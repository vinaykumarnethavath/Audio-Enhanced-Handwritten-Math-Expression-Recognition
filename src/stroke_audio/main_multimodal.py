import os, sys, re, zipfile, warnings, subprocess, importlib.util, random, shutil, difflib
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from pathlib import Path
import keras

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

if importlib.util.find_spec("librosa") is None:
    print("📦 Installing librosa...")
    install("librosa")

import librosa
from IPython.display import Audio, display

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
print(f"✅ TensorFlow Version: {tf.__version__}")

#  CONFIGURATION 
MAX_STROKE = 450
MAX_AUDIO  = 300
MFCC_DIM   = 26
MAX_TEXT   = 150

EMBED_DIM  = 256
NUM_HEADS  = 4
FF_DIM     = 512
NUM_LAYERS = 4   
DROPOUT    = 0.2

BATCH_SIZE = 64
EPOCHS     = 10
PAD, START, END = "<pad>", "<start>", "<end>"
ns = {'ink': 'http://www.w3.org/2003/InkML'}

print("✅ Configuration Loaded.")
# AUGMENTATION 
def augment_stroke_tf(stroke, label):
    scale = tf.random.uniform([], 0.85, 1.15)
    theta = tf.random.uniform([], -0.17, 0.17)
    c, s = tf.cos(theta), tf.sin(theta)
    rot_mat = tf.stack([[c, -s], [s, c]])
    coords = stroke[:, :2]
    coords = tf.matmul(coords, rot_mat) * scale
    augmented = tf.concat([coords, stroke[:, 2:3]], axis=1)
    noise = tf.random.normal(tf.shape(augmented), mean=0.0, stddev=0.01)
    return augmented + noise, label

# NORMALIZATION
def normalize_strokes(pts):
    if len(pts) == 0: return np.zeros((MAX_STROKE, 3), dtype='float32')
    arr = np.array(pts, dtype='float32')
    arr[:, :2] -= np.mean(arr[:, :2], axis=0)
    scale = np.std(arr[:, :2]) + 1e-6
    arr[:, :2] /= scale
    if arr[-1, 2] > arr[0, 2]:
        arr[:, 2] = (arr[:, 2] - arr[0, 2]) / (arr[-1, 2] - arr[0, 2])
    else: arr[:, 2] = 0.0
    if len(arr) > MAX_STROKE: return arr[:MAX_STROKE]
    return np.vstack([arr, np.zeros((MAX_STROKE - len(arr), 3), dtype='float32')])
# PARSERS 
def parse_inkml(path):
    try:
        root = ET.parse(path).getroot()
        label = ""
        for tag in ["normalizedLabel", "label"]:
            ann = root.find(f'.//{{http://www.w3.org/2003/InkML}}annotation[@type="{tag}"]')
            if ann is not None and ann.text: label = ann.text.strip(); break
        if not label: return None, None

        pts = []
        for tr in root.findall(".//ink:trace", ns):
            if tr.text:
                data = [list(map(float, p.split())) for p in tr.text.strip().split(',')]
                for p in data:
                    t = p[2] if len(p) > 2 else (0.0 if not pts else pts[-1][2] + 0.1)
                    pts.append([p[0], p[1], t])
        return normalize_strokes(pts), label
    except: return None, None

def process_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM).T
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        if mfcc.shape[0] > MAX_AUDIO: return mfcc[:MAX_AUDIO]
        return np.vstack([mfcc, np.zeros((MAX_AUDIO - mfcc.shape[0], MFCC_DIM), dtype='float32')])
    except: return np.zeros((MAX_AUDIO, MFCC_DIM), dtype='float32')

print("✅ Parsers & Helpers Ready.")
# GOOGLE DRIVE LOGIC 
if 'google.colab' in sys.modules:
    from google.colab import drive
    print("📂 Mounting Google Drive...")
    drive.mount('/content/drive')

    drive_source = Path("/content/drive/My Drive/50kaudio.zip")
    local_zip = Path("dataset.zip")

    if drive_source.exists():
        print(f"⬇️ Found file at: {drive_source}")
        print("⬇️ Copying to local runtime (this makes training much faster)...")
        shutil.copy(drive_source, local_zip)
        zip_path = str(local_zip)
    else:
        print(f"❌ ERROR: File not found at {drive_source}")
        # Fallback to manual upload
        from google.colab import files
        uploaded = files.upload()
        zip_path = list(uploaded.keys())[0]
else:
    zip_path = "dataset.zip"

# EXTRACT
extract_dir = Path(zip_path).stem + "_data"
if not os.path.exists(extract_dir):
    print("📂 Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_dir)
    except zipfile.BadZipFile:
        print("❌ Corrupted Zip")
else:
    print("📂 Data already extracted.")

print("✅ Data Ready on Local Disk.")
def build_dataset(extracted_path):
    all_files = list(Path(extracted_path).rglob("*.inkml"))
    print(f"🔎 Found {len(all_files)} files. Loading into memory...")

    X_s, X_a, Y_raw, Meta_paths = [], [], [], []

    for f in all_files:
        s, l = parse_inkml(f)
        if s is not None:
            mp3 = f.parent / f"{f.stem}.mp3"
            a = process_audio(mp3) if mp3.exists() else np.zeros((MAX_AUDIO, MFCC_DIM), dtype='float32')

            X_s.append(s)
            X_a.append(a)
            Y_raw.append(l)
            Meta_paths.append(str(f))

    if not X_s:
        print("❌ No valid data loaded."); return None, None, None, None

    # Tokenizer
    vocab = {PAD: 0, START: 1, END: 2}
    inv_vocab = {0: PAD, 1: START, 2: END}
    vec_labels = []

    for l in Y_raw:
        toks = re.findall(r"\\[a-zA-Z]+|[^ ]", l)
        ids = [1]
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
                inv_vocab[len(vocab)-1] = t
            ids.append(vocab[t])
        ids.append(2)
        ids = ids[:MAX_TEXT] + [0]*(MAX_TEXT - len(ids))
        vec_labels.append(ids)

    # Convert to Numpy
    X_s = np.array(X_s, dtype='float32')
    X_a = np.array(X_a, dtype='float32')
    Y   = np.array(vec_labels, dtype='int32')
    M_p = np.array(Meta_paths)

    idx = np.random.permutation(len(Y))
    sp = int(len(Y) * 0.9)
    train_idx, val_idx = idx[:sp], idx[sp:]

    ds_train = tf.data.Dataset.from_tensor_slices((
        {"stroke": X_s[train_idx], "audio": X_a[train_idx], "tok": Y[train_idx][:, :-1], "fpath": M_p[train_idx]},
        Y[train_idx][:, 1:]
    ))

    def map_aug(inputs, targets):
        inputs["stroke"], _ = augment_stroke_tf(inputs["stroke"], targets)
        return inputs, targets

    ds_train = ds_train.shuffle(2000).map(map_aug, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((
        {"stroke": X_s[val_idx], "audio": X_a[val_idx], "tok": Y[val_idx][:, :-1], "fpath": M_p[val_idx]},
        Y[val_idx][:, 1:]
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, vocab, inv_vocab

# EXECUTE BUILDER
train_ds, val_ds, vocab, inv_vocab = build_dataset(extract_dir)
if vocab: print(f"✅ Dataset Built. Vocab Size: {len(vocab)}")
# LAYERS 
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
    def call(self, x): return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    def get_config(self):
        c = super().get_config(); c.update({"max_len": self.max_len, "d_model": self.d_model}); return c

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.dense_dim = embed_dim, dense_dim
        self.num_heads, self.dropout_rate = num_heads, dropout_rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([keras.layers.Dense(dense_dim, activation="relu"), keras.layers.Dense(embed_dim)])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        c = super().get_config(); c.update({"embed_dim": self.embed_dim, "dense_dim": self.dense_dim, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate}); return c

class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.dense_dim = embed_dim, dense_dim
        self.num_heads, self.dropout_rate = num_heads, dropout_rate
        self.att1 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([keras.layers.Dense(dense_dim, activation="relu"), keras.layers.Dense(embed_dim)])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dropout3 = keras.layers.Dropout(dropout_rate)
    def call(self, inputs, encoder_outputs, training=False):
        attn1 = self.att1(inputs, inputs, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        attn2 = self.att2(out1, encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.layernorm3(out2 + ffn_out)
    def get_config(self):
        c = super().get_config(); c.update({"embed_dim": self.embed_dim, "dense_dim": self.dense_dim, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate}); return c

def build_transformer(vocab_len):
    # 1. Stroke Encoder
    s_in = keras.Input(shape=(MAX_STROKE, 3), name="stroke")
    s_emb = keras.layers.Dense(EMBED_DIM)(s_in)
    s_emb = PositionalEncoding(MAX_STROKE, EMBED_DIM)(s_emb)
    s_x = s_emb
    for _ in range(NUM_LAYERS):
        s_x = TransformerEncoder(EMBED_DIM, FF_DIM, NUM_HEADS, DROPOUT)(s_x)

    # 2. Audio Encoder
    a_in = keras.Input(shape=(MAX_AUDIO, MFCC_DIM), name="audio")
    a_emb = keras.layers.Dense(EMBED_DIM)(a_in)
    a_emb = PositionalEncoding(MAX_AUDIO, EMBED_DIM)(a_emb)
    a_x = a_emb
    for _ in range(NUM_LAYERS):
        a_x = TransformerEncoder(EMBED_DIM, FF_DIM, NUM_HEADS, DROPOUT)(a_x)

    # 3. Fusion
    context = keras.layers.Concatenate(axis=1)([s_x, a_x])

    # 4. Decoder
    tok_in = keras.Input(shape=(MAX_TEXT-1,), name="tok")
    dec_emb = keras.layers.Embedding(vocab_len, EMBED_DIM)(tok_in)
    dec_emb = PositionalEncoding(MAX_TEXT-1, EMBED_DIM)(dec_emb)
    x = dec_emb
    for _ in range(NUM_LAYERS):
        x = TransformerDecoder(EMBED_DIM, FF_DIM, NUM_HEADS, DROPOUT)(x, context)

    outputs = keras.layers.Dense(vocab_len)(x)
    return keras.Model(inputs=[s_in, a_in, tok_in], outputs=outputs)

print("✅ Model Architecture Defined.")
# Learning Rate Schedule
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": float(self.d_model), "warmup_steps": self.warmup_steps}
# Dataset Filter

def filter_for_training(inputs, labels):
    return {
        "stroke": inputs["stroke"],
        "audio": inputs["audio"],
        "tok": inputs["tok"]
    }, labels
# MAIN LOGIC

if vocab:

    print("🧹 Creating clean datasets for GPU training...")
    train_ds_clean = train_ds.map(filter_for_training, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_clean   = val_ds.map(filter_for_training, num_parallel_calls=tf.data.AUTOTUNE)

    model = build_transformer(len(vocab))

    lr_schedule = CustomSchedule(EMBED_DIM)
    opt = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    # CHECKPOINT SETUP
    if 'google.colab' in sys.modules:
        ckpt_dir = Path("/content/drive/My Drive/Colab_Checkpoints")
    else:
        ckpt_dir = Path("checkpoints")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.weights.h5"
    # LOAD OR DECIDE
    if ckpt_path.exists():
        print(f"\n✅ Found checkpoint: {ckpt_path}")
        choice = input("Enter 't' to TRAIN MORE or 'e' to EVALUATE only: ").strip().lower()

        model.load_weights(str(ckpt_path))
        print("✅ Weights loaded.")

    else:
        print("\n✨ No checkpoint found. Starting fresh.")
        choice = 't'
    # TRAINING
    if choice == 't':

        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        print("🔥 Starting Training...\n")

        history = model.fit(
            train_ds_clean,
            validation_data=val_ds_clean,
            epochs=EPOCHS,
            callbacks=[checkpoint_cb, early_stopping]
        )

        print("✅ Training Completed.")


else:
    print("❌ Dataset not ready. Please run dataset preparation cell.")
def decode_sequence(model, s, a, i2t, t2i):
    curr = np.zeros((1, MAX_TEXT-1), dtype='int32')
    curr[0,0] = t2i[START]
    for i in range(MAX_TEXT-2):
        pred = model.predict({"stroke": s, "audio": a, "tok": curr}, verbose=0)
        nxt = np.argmax(pred[0, i, :])
        if nxt == t2i[END]: break
        curr[0, i+1] = nxt
    return "".join([i2t[x] for x in curr[0] if x not in [0, 1, 2]])

def calculate_metrics(model, dataset, vocab, inv_vocab, num_samples=100):
    print(f"\n📊 Calculating accuracy on {num_samples} validation samples...")
    expr_perfect = 0
    total_tokens = 0
    correct_tokens = 0
    processed = 0

    for batch in dataset:
        inputs, true_y = batch
        bs = tf.shape(true_y)[0]
        for i in range(bs):
            if processed >= num_samples: break

            s_sample, a_sample = inputs["stroke"][i:i+1], inputs["audio"][i:i+1]
            pred_str = decode_sequence(model, s_sample, a_sample, inv_vocab, vocab)

            true_seq = true_y[i].numpy()
            true_str = "".join([inv_vocab[x] for x in true_seq if x in inv_vocab and x not in [0, 1, 2]])

            if pred_str == true_str: expr_perfect += 1

            pred_toks = re.findall(r"\\[a-zA-Z]+|[^ ]", pred_str)
            true_toks = re.findall(r"\\[a-zA-Z]+|[^ ]", true_str)
            matcher = difflib.SequenceMatcher(None, true_toks, pred_toks)
            matches = sum(triple[-1] for triple in matcher.get_matching_blocks())
            correct_tokens += matches
            total_tokens += max(len(true_toks), len(pred_toks))
            processed += 1
            if processed % 100 == 0: print(f"   Processed {processed}/{num_samples}...")
        if processed >= num_samples: break

    print(f"\n🏆 Expression Accuracy: {(expr_perfect/processed)*100:.2f}%")
    print(f"🔠 Token Accuracy:      {(correct_tokens/max(total_tokens,1))*100:.2f}%")
if vocab:
    calculate_metrics(model, val_ds, vocab, inv_vocab, num_samples=500)

#  VISUALIZATION 
def visualize_with_audio(model, dataset, i2t, t2i, num_samples=5):
    print("\n" + "="*80)
    print(f"🎧 AUDIO VISUALIZATION & INSPECTION ({num_samples} Samples)")
    print("="*80)

    for batch in dataset.take(1):
        inputs, true_labels = batch
        limit = min(num_samples, tf.shape(true_labels)[0])
        for i in range(limit):
        
            ink_path_bytes = inputs["fpath"][i].numpy()
            ink_path_str = ink_path_bytes.decode("utf-8")
            path_obj = Path(ink_path_str)
            audio_path = path_obj.with_suffix('.mp3')
            folder_name = path_obj.parent.name
            ink_filename = path_obj.name

            # 2. Extract Data
            s_sample = inputs["stroke"][i:i+1]
            a_sample = inputs["audio"][i:i+1]

            # 3. Generate Prediction
            curr = np.zeros((1, MAX_TEXT-1), dtype='int32')
            curr[0,0] = t2i[START]
            for t in range(MAX_TEXT-2):
                pred = model.predict({"stroke": s_sample, "audio": a_sample, "tok": curr}, verbose=0)
                nxt = np.argmax(pred[0, t, :])
                if nxt == t2i[END]: break
                curr[0, t+1] = nxt
            pred_str = "".join([i2t[x] for x in curr[0] if x not in [0, 1, 2]])

            # 4. Get Real Label
            true_seq = true_labels[i].numpy()
            true_str = "".join([i2t[x] for x in true_seq if x in i2t and x not in [0, 1, 2]])

            # 5. Display
            print(f"Sample #{i+1} | Folder: {folder_name} | File: {ink_filename}")
            if audio_path.exists():
                display(Audio(filename=str(audio_path)))
            else:
                print(f"⚠️ Audio file missing at: {audio_path}")

            print(f"🟢 Real:      {true_str}")
            print(f"🤖 Predicted: {pred_str}")
if vocab:
    visualize_with_audio(model, val_ds, inv_vocab, vocab, num_samples=10)
