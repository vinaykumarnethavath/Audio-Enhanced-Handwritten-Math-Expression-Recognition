
import importlib.util, subprocess, sys, warnings, os, re, json, heapq
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np, tensorflow as tf, zipfile
import keras

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Checking dependencies...")
if importlib.util.find_spec("tensorflow") is None:
    print("TensorFlow not found → installing now..."); _pip("tensorflow")

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
try:
    res = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(res); tf.tpu.experimental.initialize_tpu_system(res)
    strat = tf.distribute.TPUStrategy(res)
    print("✅ TPU ready")
except:
    strat = tf.distribute.get_strategy()
    print("⚠️ TPU not found – using", strat.__class__.__name__)
MAX_STROKE = 500
MAX_TOKENS = 100
EMBED_DIM  = 256
NUM_HEADS  = 4
NUM_LAYERS = 4
FF_DIM     = 1024
BATCH_SIZE = 32
EPOCHS     = 10
TEST_SPLIT = 0.10
RNG_SEED   = 42

PAD, START, END = "[PAD]", "[START]", "[END]"
_tok_re = re.compile(r"\\[A-Za-z]+|[^ ]")
ns = {'ink':'http://www.w3.org/2003/InkML'}

# ---- TOKENISER ----
def tokenize(expr):
    return _tok_re.findall(expr)

def build_tokeniser(labels):
    t2i = {PAD: 0, START: 1, END: 2}
    i2t = {0: PAD, 1: START, 2: END}
    nxt = 3
    seqs = []

    for lab in labels:
        toks = tokenize(lab)
        seqs.append(toks)
        for t in toks:
            if t not in t2i:
                t2i[t] = nxt
                i2t[nxt] = t
                nxt += 1

    return t2i, i2t, seqs

def encode(seqs, t2i):
    dec_in, dec_tar = [], []

    for toks in seqs:
        ids = [t2i[START]] + [t2i[t] for t in toks] + [t2i[END]]
        ids = ids[:MAX_TOKENS]
        ids += [t2i[PAD]] * (MAX_TOKENS - len(ids))
        dec_in.append(ids[:-1])
        dec_tar.append(ids[1:])

    return np.asarray(dec_in, np.int32), np.asarray(dec_tar, np.int32)

# ---- INKML PARSER -----
def parse_inkml(path: Path):
    """Return strokes (N,3) with Δt & global normalisation + label str."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None, None

    root = tree.getroot()
    pts = []; last_t = 0.0
    for tr in root.findall(".//ink:trace", ns):
        if tr.text is None: continue
        for seg in tr.text.strip().replace('\n', '').split(','):
            seg = seg.strip()
            if not seg: continue
            parts = seg.split()
            if len(parts) == 3:
                x, y, t = map(float, parts); last_t = t
            else:
                x, y = map(float, parts[:2]); t = last_t + 1.0
            pts.append((x, y, t))

    pts = np.asarray(pts, np.float32)
    if pts.size == 0: pts = np.zeros((1, 3), np.float32)

    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    sx, sy = x.min(), y.min()
    scale = max(x.max() - sx, y.max() - sy, 1e-6)
    pts[:, 0] = (x - sx) / scale
    pts[:, 1] = (y - sy) / scale
    pts[:, 2] = 0 if t.max() == t.min() else (t - t.min()) / (t.max() - t.min())

    if len(pts) > MAX_STROKE: pts = pts[:MAX_STROKE]
    if len(pts) < MAX_STROKE:
        pts = np.vstack([pts, np.zeros((MAX_STROKE - len(pts), 3), np.float32)])

    latex = ""
    for tag in ("normalizedLabel", "label"):
        ann = root.find(f'.//{{http://www.w3.org/2003/InkML}}annotation[@type="{tag}"]')
        if ann is not None and ann.text:
            latex = ann.text.strip()
            break
    return pts, latex

# ----- DATASET PIPE -----
def prepare_dataset(data_path: Path):
    if data_path.is_file() and data_path.suffix == '.zip':
        zip_path = data_path
        unzip_dir = data_path.parent / data_path.stem
        if not unzip_dir.is_dir():
            print(f"🔄 Unzipping dataset from '{zip_path}' to '{unzip_dir}'...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)
            except zipfile.BadZipFile:
                raise RuntimeError(f"🚫 ERROR: Corrupted zip file at '{zip_path}'.")
        data_path = unzip_dir

    if not data_path.is_dir():
            raise FileNotFoundError(f"Dataset folder not found at '{data_path}'.")

    strokes, labels = [], []
    inkml_files = list(data_path.glob('**/*.inkml'))

    if not inkml_files:
        raise RuntimeError(f"No .inkml files found within '{data_path}'.")

    print(f"Found {len(inkml_files)} InkML files. Processing...")

    for ink_path in sorted(inkml_files):
        try:
            s, l = parse_inkml(ink_path)
            if s is not None and l: 
                strokes.append(s)
                labels.append(l)
        except Exception as e:
            print(f"⚠️ Error parsing '{ink_path.stem}': {e}")
            continue

    if not strokes:
        raise RuntimeError("No valid data found in the dataset directory.")

    strokes = np.stack(strokes)
    t2i, i2t, seqs = build_tokeniser(labels)
    dec_in, dec_tar = encode(seqs, t2i)

    w = (dec_tar != t2i[PAD]).astype(np.float32)

    n = len(strokes)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(n)
    split = int(n * TEST_SPLIT)
    train_idx, val_idx = idx[split:], idx[:split]

    # Notice: Audio input removed from dataset structure
    def tfds(ix):
        return tf.data.Dataset.from_tensor_slices(
            ((strokes[ix], dec_in[ix]), dec_tar[ix], w[ix])
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return tfds(train_idx), tfds(val_idx), strokes, labels, t2i, i2t

# ---- MODEL ------
class PosEnc(tf.keras.layers.Layer):
    def __init__(self, length, dim):
        super().__init__()
        self.pe = keras.layers.Embedding(length, dim)

    def call(self, x):
        return x + self.pe(tf.range(tf.shape(x)[1])[tf.newaxis, :])

def build_model(vocab):
    st_in = keras.Input((MAX_STROKE, 3), name="stroke")
    tok_in = keras.Input((MAX_TOKENS - 1,), dtype=tf.int32, name="tok")

    #  STROKE ENCODER 
    st_mask = keras.layers.Lambda(lambda t: keras.ops.any(keras.ops.not_equal(t, 0.), axis=-1))(st_in)

    s = keras.layers.Dense(EMBED_DIM)(st_in)
    s = PosEnc(MAX_STROKE, EMBED_DIM)(s)
    s_proj = keras.layers.Dense(EMBED_DIM // NUM_HEADS)(s)

    for i in range(NUM_LAYERS):
        att = keras.layers.MultiHeadAttention(NUM_HEADS, EMBED_DIM // NUM_HEADS, dropout=0.1)(
            s_proj, s_proj, s_proj, attention_mask=st_mask[:, tf.newaxis, :])
        s_proj = keras.layers.LayerNormalization(epsilon=1e-6)(s_proj + att)
        ffn = keras.layers.Dense(FF_DIM, activation='relu')(s_proj)
        ffn = keras.layers.Dense(EMBED_DIM // NUM_HEADS)(ffn)
        s_proj = keras.layers.LayerNormalization(epsilon=1e-6)(s_proj + ffn)

    enc_out = s_proj

    # --- DECODER ---
    y = keras.layers.Embedding(vocab, EMBED_DIM, mask_zero=True)(tok_in)
    y = PosEnc(MAX_TOKENS, EMBED_DIM)(y)
    y_proj = keras.layers.Dense(EMBED_DIM // NUM_HEADS)(y)

    cross_attention_mask = st_mask[:, tf.newaxis, tf.newaxis, :]

    for i in range(NUM_LAYERS):
     
        self_att = keras.layers.MultiHeadAttention(NUM_HEADS, EMBED_DIM // NUM_HEADS, dropout=0.1)(
            y_proj, y_proj, y_proj, use_causal_mask=True)
        y_proj = keras.layers.LayerNormalization(epsilon=1e-6)(y_proj + self_att)

        cross = keras.layers.MultiHeadAttention(NUM_HEADS, EMBED_DIM // NUM_HEADS, dropout=0.1)(
            y_proj, enc_out, enc_out, attention_mask=cross_attention_mask)
        y_proj = keras.layers.LayerNormalization(epsilon=1e-6)(y_proj + cross)

        ffn = keras.layers.Dense(FF_DIM, activation='relu')(y_proj)
        ffn = keras.layers.Dense(EMBED_DIM // NUM_HEADS)(ffn)
        y_proj = keras.layers.LayerNormalization(epsilon=1e-6)(y_proj + ffn)

    logits = keras.layers.Dense(vocab)(keras.layers.Dense(EMBED_DIM)(y_proj))

    def m_acc(y_true, y_pred):
        mask = keras.ops.cast(keras.ops.not_equal(y_true, 0), tf.float32)
        predicted_ids = keras.ops.argmax(y_pred, axis=-1)
        predicted_ids = keras.ops.cast(predicted_ids, y_true.dtype)
        match = keras.ops.cast(keras.ops.equal(predicted_ids, y_true), tf.float32)
        return keras.ops.sum(mask * match) / keras.ops.sum(mask)

    model = keras.Model([st_in, tok_in], logits)
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[m_acc]
    )
    return model

# ---- DECODERS -----
def greedy_decode(model, stroke, t2i, i2t):
    seq = np.full((1, MAX_TOKENS - 1), t2i[PAD], np.int32)
    seq[0, 0] = t2i[START]

    for t in range(MAX_TOKENS - 1):
        # Predict using only Stroke + Sequence
        logits = model.predict([stroke, seq], verbose=0)[0, t]
        nxt = int(np.argmax(logits))

        if nxt == t2i[END]:
            break

        if t + 1 < MAX_TOKENS - 1:
            seq[0, t + 1] = nxt

    return "".join(i2t[i] for i in seq[0] if i not in (t2i[PAD], t2i[START], t2i[END]))

def beam_decode(model, stroke, t2i, i2t, width=5):
    beams = [(0.0, [t2i[START]])]

    for _ in range(MAX_TOKENS - 1):
        next_beams = []
        for score, seq in beams:
            if seq[-1] == t2i[END]:
                next_beams.append((score, seq))
                continue

            inp = np.full((1, MAX_TOKENS - 1), t2i[PAD], np.int32)
            inp[0, :len(seq)] = seq

            # Predict using only Stroke + Sequence
            logits = model.predict([stroke, inp], verbose=0)[0, len(seq) - 1]
            logp = tf.nn.log_softmax(logits).numpy()

            for idx in np.argsort(logp)[-width:]:
                next_beams.append((score + logp[idx], seq + [int(idx)]))

        beams = heapq.nlargest(width, next_beams, key=lambda x: x[0])

    best = max(beams, key=lambda x: x[0])[1]
    return "".join(i2t[i] for i in best if i not in (t2i[PAD], t2i[START], t2i[END]))

def evaluate(model, strokes, labels, idx, t2i, i2t):
    expr_ok = 0
    tok_ok = 0
    tok_tot = 0

    for i in idx:
        pred = greedy_decode(model, strokes[i:i + 1], t2i, i2t)
        true = labels[i]

        if pred == true:
            expr_ok += 1

        t_true, t_pred = tokenize(true), tokenize(pred)
        tok_tot += max(len(t_true), len(t_pred))
        tok_ok += sum(1 for a, b in zip(t_true, t_pred) if a == b)

    print(f"Expression accuracy: {expr_ok / len(idx):.2%}")
    print(f"Token accuracy     : {tok_ok / max(1, tok_tot):.2%}")

# ----- MAIN ----
if __name__ == "__main__":

    if 'google.colab' in sys.modules:
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            print("🚫 ERROR: No file uploaded.")
            sys.exit(1)
        dataset_filename = list(uploaded.keys())[0]
        dataset_path = Path(dataset_filename)
    else:

        dataset_path = Path("./dataset.zip")

    class MockArgs:
        def __init__(self, data_path, predict=None):
            self.data_path = data_path
            self.predict = predict

    args = MockArgs(data_path=dataset_path)

    try:
        train_ds, val_ds, strokes, labels, t2i, i2t = prepare_dataset(args.data_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"🚫 ERROR: {e}")
        sys.exit(1)

    if args.predict:
        model_path = Path("./stroke_only_model.h5")
        if not model_path.exists():
            print(f"🚫 ERROR: Model file not found at '{model_path}'.")
            sys.exit(1)

        with strat.scope():
            model = tf.keras.models.load_model(model_path, compile=False)

        if args.predict.isdigit():
            idx = int(args.predict) - 1
            if not 0 <= idx < len(strokes):
                print(f"🚫 ERROR: Sample ID {args.predict} is out of range.")
                sys.exit(1)
            stroke = strokes[idx:idx + 1]
        else:
            inkml_file_path = Path(args.predict)
            if not inkml_file_path.exists():
                print(f"🚫 ERROR: InkML file not found at '{inkml_file_path.resolve()}'")
                sys.exit(1)
            stroke, _ = parse_inkml(inkml_file_path)
            stroke = stroke[np.newaxis]

        print(beam_decode(model, stroke, t2i, i2t))
        sys.exit(0)

    #  Training Mode 
    with strat.scope():
        model = build_model(len(t2i))

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4)
    ]

    print("\nStarting Training (InkML Only)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\n🧮 Validation evaluation")
    v_idx = np.arange(int(len(strokes) * TEST_SPLIT))
    evaluate(model, strokes, labels, v_idx, t2i, i2t)

    model.save("./stroke_only_model.h5")
    with open("./stroke_tokenizer.json", "w") as f:
        json.dump({"vocab": [i2t[i] for i in range(len(i2t))]}, f)
    print("✅ Saved model & tokenizer to current directory")
