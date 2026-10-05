

#!pip install -q datasets pythainlp scikit-learn


# ============================================================
# 1. IMPORT
# ============================================================

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from pythainlp.tokenize import word_tokenize

import tensorflow as tf
import keras

from keras import layers
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.model_selection import train_test_split


# ============================================================
# 2. CONFIG
# ============================================================

NUM_DOCUMENTS = 3000

MAX_FEATURES = 10000

SEQ_LENGTH = 10

MAX_SEQUENCES = 300000

EMBEDDING_DIM = 64

LSTM_UNITS = 64

BATCH_SIZE = 256

EPOCHS = 15

RANDOM_SEED = 42

SAVE_DIR = "thai_next_word_model"

TFLITE_PATH = "thai_next_word_model.tflite"

VOCAB_PATH = "vocabulary.txt"


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# 3. GPU CHECK
# ============================================================

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected!")

    for gpu in gpus:
        print(gpu)

else:

    print("WARNING: GPU NOT DETECTED")

print()


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING THAI WIKIPEDIA")
print("=" * 60)

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.th",
    split=f"train[:{NUM_DOCUMENTS}]"
)

df = dataset.to_pandas()

print(
    "Documents:",
    len(df)
)

print()


# ============================================================
# 5. CLEAN TEXT
# ============================================================

print("=" * 60)
print("CLEANING TEXT")
print("=" * 60)


def clean_thai_text(text):

    text = str(text)

    # Remove URL
    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    # Remove HTML
    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    # Keep Thai / English / Number / Space
    text = re.sub(
        r'[^\u0E00-\u0E7F a-zA-Z0-9\s]',
        '',
        text
    )

    # Remove multiple spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


df["cleaned_text"] = df["text"].apply(
    clean_thai_text
)


print("Cleaning complete.")
print()


# ============================================================
# 6. THAI TOKENIZATION
# ============================================================

print("=" * 60)
print("THAI TOKENIZATION")
print("=" * 60)


def tokenize_thai(text):

    tokens = word_tokenize(
        text,
        engine="newmm",
        keep_whitespace=False
    )

    tokens = [
        token.strip()
        for token in tokens
        if token.strip()
    ]

    return " ".join(tokens)


df["processed_text"] = df["cleaned_text"].apply(
    tokenize_thai
)


# Remove short documents

df = df[
    df["processed_text"]
    .str.split()
    .str.len()
    >= SEQ_LENGTH + 1
].reset_index(drop=True)


print(
    "Usable documents:",
    len(df)
)

print()


# ============================================================
# 7. TEXT VECTORIZATION
# ============================================================

print("=" * 60)
print("BUILDING VOCABULARY")
print("=" * 60)


vectorize_layer = layers.TextVectorization(

    max_tokens=MAX_FEATURES,

    output_mode="int",

    split="whitespace"
)


vectorize_layer.adapt(
    df["processed_text"].values
)


vocab = vectorize_layer.get_vocabulary()

VOCAB_SIZE = len(vocab)


print(
    "Vocabulary size:",
    VOCAB_SIZE
)

print(
    "First 20 words:"
)

print(
    vocab[:20]
)

print()


# ============================================================
# 8. CREATE TRAINING SEQUENCES
# ============================================================

print("=" * 60)
print("CREATING TRAINING SEQUENCES")
print("=" * 60)


sequences = []


for doc_index, text in enumerate(
    df["processed_text"]
):

    tokens = text.split()

    # Convert tokens to integer IDs

    ids = vectorize_layer(
        [" ".join(tokens)]
    ).numpy()[0]


    # Remove padding

    ids = ids[
        ids != 0
    ]


    if len(ids) <= SEQ_LENGTH:
        continue


    # Number of possible positions

    possible_positions = (
        len(ids) - SEQ_LENGTH
    )


    # Limit sequences per document

    samples_per_doc = min(
        100,
        possible_positions
    )


    # Random positions

    positions = np.random.choice(

        possible_positions,

        size=samples_per_doc,

        replace=False
    )


    for pos in positions:

        seq = ids[
            pos:pos + SEQ_LENGTH + 1
        ]

        sequences.append(seq)


        if len(sequences) >= MAX_SEQUENCES:
            break


    if len(sequences) >= MAX_SEQUENCES:
        break


    if doc_index % 500 == 0:

        print(
            f"Documents: {doc_index:,} | "
            f"Sequences: {len(sequences):,}"
        )


sequences = np.array(
    sequences,
    dtype=np.int32
)


print()

print(
    "Total sequences:",
    len(sequences)
)

print(
    "Sequence shape:",
    sequences.shape
)

print()


# ============================================================
# 9. X / Y
# ============================================================

X = sequences[:, :-1]

y = sequences[:, -1]


print("=" * 60)
print("DATA SHAPE")
print("=" * 60)

print(
    "X:",
    X.shape
)

print(
    "y:",
    y.shape
)

print()


# ============================================================
# 10. TRAIN / VALIDATION SPLIT
# ============================================================

print("=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)


X_train, X_val, y_train, y_val = train_test_split(

    X,

    y,

    test_size=0.1,

    random_state=RANDOM_SEED
)


print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print()


# ============================================================
# 11. TF.DATA PIPELINE
# ============================================================

print("=" * 60)
print("BUILDING TF.DATA PIPELINE")
print("=" * 60)


train_ds = tf.data.Dataset.from_tensor_slices(

    (
        X_train,
        y_train
    )
)


train_ds = train_ds.shuffle(

    10000,

    seed=RANDOM_SEED
)


train_ds = train_ds.batch(
    BATCH_SIZE
)


train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


val_ds = tf.data.Dataset.from_tensor_slices(

    (
        X_val,
        y_val
    )
)


val_ds = val_ds.batch(
    BATCH_SIZE
)


val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


print("Dataset ready.")
print()


# ============================================================
# 12. BUILD MODEL
# ============================================================

print("=" * 60)
print("BUILDING MODEL")
print("=" * 60)


model = keras.Sequential([

    layers.Embedding(

        input_dim=VOCAB_SIZE,

        output_dim=EMBEDDING_DIM,

        name="embedding"
    ),


    layers.LSTM(

        LSTM_UNITS,

        name="lstm"
    ),


    layers.Dropout(

        0.2,

        name="dropout"
    ),


    layers.Dense(

        128,

        activation="relu",

        name="dense"
    ),


    layers.Dropout(

        0.2,

        name="dense_dropout"
    ),


    layers.Dense(

        VOCAB_SIZE,

        activation="softmax",

        name="output"
    )

])


# ============================================================
# 13. COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(

        learning_rate=0.001
    ),


    loss="sparse_categorical_crossentropy",


    metrics=[

        keras.metrics.SparseCategoricalAccuracy(

            name="accuracy"
        ),


        keras.metrics.SparseTopKCategoricalAccuracy(

            k=5,

            name="top5_accuracy"
        )

    ]
)


model.summary()


# ============================================================
# 14. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=1,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# 15. TRAIN
# ============================================================

print()
print("=" * 60)
print("START TRAINING")
print("=" * 60)
print()


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[

        early_stopping,

        reduce_lr

    ],

    verbose=1
)


# ============================================================
# 16. EVALUATE
# ============================================================

print()
print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)


results = model.evaluate(

    val_ds,

    verbose=1
)


for metric_name, value in zip(

    model.metrics_names,

    results
):

    print(
        f"{metric_name}: {value:.4f}"
    )


print()


# ============================================================
# 17. TRAINING GRAPHS
# ============================================================

print("=" * 60)
print("TRAINING GRAPHS")
print("=" * 60)


history_dict = history.history


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history_dict["accuracy"],

    label="Training Accuracy"
)


plt.plot(

    history_dict["val_accuracy"],

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Thai Next Word Prediction - Accuracy"
)


plt.legend()


plt.grid()


plt.show()


# ------------------------------------------------------------
# Top-5 Accuracy
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history_dict["top5_accuracy"],

    label="Training Top-5"
)


plt.plot(

    history_dict["val_top5_accuracy"],

    label="Validation Top-5"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Top-5 Accuracy"
)


plt.title(
    "Thai Next Word Prediction - Top-5 Accuracy"
)


plt.legend()


plt.grid()


plt.show()


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history_dict["loss"],

    label="Training Loss"
)


plt.plot(

    history_dict["val_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Thai Next Word Prediction - Loss"
)


plt.legend()


plt.grid()


plt.show()


# ============================================================
# 18. SAVE KERAS MODEL
# ============================================================

print("=" * 60)
print("SAVING KERAS MODEL")
print("=" * 60)


keras_path = "thai_next_word_model.keras"


model.save(
    keras_path
)


print(
    "Keras model saved:"
)

print(
    keras_path
)

print()


# ============================================================
# 19. SAVE TENSORFLOW SAVEDMODEL
# ============================================================

print("=" * 60)
print("EXPORTING TENSORFLOW SAVEDMODEL")
print("=" * 60)


# Remove old directory if needed

if os.path.exists(SAVE_DIR):

    import shutil

    shutil.rmtree(
        SAVE_DIR
    )


# Export SavedModel

model.export(
    SAVE_DIR
)


print(
    "TensorFlow SavedModel saved:"
)

print(
    SAVE_DIR
)

print()


# ============================================================
# 20. SAVE VOCABULARY
# ============================================================

print("=" * 60)
print("SAVING VOCABULARY")
print("=" * 60)


with open(

    VOCAB_PATH,

    "w",

    encoding="utf-8"

) as f:

    for word in vocab:

        f.write(
            word + "\n"
        )


print(
    "Vocabulary saved:"
)

print(
    VOCAB_PATH
)

print()


# ============================================================
# 21. SAVE CONFIG
# ============================================================

config = {

    "max_features": MAX_FEATURES,

    "seq_length": SEQ_LENGTH,

    "embedding_dim": EMBEDDING_DIM,

    "lstm_units": LSTM_UNITS,

    "vocab_size": VOCAB_SIZE

}


import json


with open(

    "model_config.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        config,

        f,

        ensure_ascii=False,

        indent=4
    )


print(
    "Config saved:"
)

print(
    "model_config.json"
)

print()


# ============================================================
# 22. CONVERT TO TFLITE
# ============================================================

print("=" * 60)
print("CONVERTING TO TFLITE")
print("=" * 60)


converter = tf.lite.TFLiteConverter.from_saved_model(

    SAVE_DIR
)


# Optimization

converter.optimizations = [

    tf.lite.Optimize.DEFAULT

]


tflite_model = converter.convert()


with open(

    TFLITE_PATH,

    "wb"

) as f:

    f.write(
        tflite_model
    )


print(
    "TFLite model saved:"
)

print(
    TFLITE_PATH
)


print(
    "TFLite size:",
    round(
        os.path.getsize(
            TFLITE_PATH
        ) / (1024 * 1024),
        2
    ),
    "MB"
)

print()


# ============================================================
# 23. TEST TFLITE MODEL
# ============================================================

print("=" * 60)
print("TESTING TFLITE MODEL")
print("=" * 60)


interpreter = tf.lite.Interpreter(

    model_path=TFLITE_PATH

)


interpreter.allocate_tensors()


input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()


print(
    "Input:"
)

print(
    input_details
)


print(
    "\nOutput:"
)

print(
    output_details
)


print(
    "\nTFLite model loaded successfully!"
)

print()


# ============================================================
# 24. NEXT WORD PREDICTION
# ============================================================

def suggest_next_words(

    text,

    top_k=5,

    temperature=0.7

):

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    cleaned = clean_thai_text(
        text
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    tokens = word_tokenize(

        cleaned,

        engine="newmm",

        keep_whitespace=False

    )


    tokens = [

        token.strip()

        for token in tokens

        if token.strip()

    ]


    if len(tokens) == 0:

        return []


    # --------------------------------------------------------
    # Last N words
    # --------------------------------------------------------

    tokens = tokens[
        -SEQ_LENGTH:
    ]


    # --------------------------------------------------------
    # Vectorize
    # --------------------------------------------------------

    text_for_vectorize = " ".join(
        tokens
    )


    ids = vectorize_layer(

        [text_for_vectorize]

    ).numpy()[0]


    ids = ids[
        ids != 0
    ]


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    if len(ids) < SEQ_LENGTH:

        ids = np.pad(

            ids,

            (
                SEQ_LENGTH - len(ids),

                0
            ),

            mode="constant"
        )

    else:

        ids = ids[
            -SEQ_LENGTH:
        ]


    X_input = np.array(

        [ids],

        dtype=np.int32

    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probs = model.predict(

        X_input,

        verbose=0

    )[0]


    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    logits = np.log(

        probs + 1e-10

    )


    logits = (

        logits /
        temperature

    )


    exp_logits = np.exp(

        logits -

        np.max(logits)

    )


    probs = (

        exp_logits /

        np.sum(exp_logits)

    )


    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    top_indices = np.argsort(

        probs

    )[::-1]


    suggestions = []


    for idx in top_indices:

        word = vocab[idx]


        if word in [

            "",

            "[UNK]",

            "[START]",

            "[END]"

        ]:

            continue


        if word.strip() == "":

            continue


        suggestions.append({

            "word": word,

            "probability": float(

                probs[idx]

            )

        })


        if len(suggestions) >= top_k:

            break


    return suggestions


# ============================================================
# 25. INTERACTIVE TEST
# ============================================================

print("=" * 60)
print("THAI NEXT WORD PREDICTION")
print("=" * 60)

print(
    "Type 'exit' to stop."
)

print()


while True:

    text = input(
        "Enter Thai sentence: "
    )


    if text.lower() == "exit":

        break


    suggestions = suggest_next_words(

        text,

        top_k=5,

        temperature=0.7

    )


    print()

    print(
        "Input:"
    )

    print(
        text
    )


    print()

    print(
        "Top 5 suggestions:"
    )


    for i, item in enumerate(

        suggestions,

        start=1

    ):

        print(

            f"{i}. "
            f"{item['word']} "
            f"("
            f"{item['probability'] * 100:.2f}%"
            f")"

        )


    print()


# ============================================================
# 26. FINAL FILE LIST
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()

print("Generated files:")

print()

print("1. Keras model:")
print(
    "   thai_next_word_model.keras"
)

print()

print("2. TensorFlow SavedModel:")
print(
    f"   {SAVE_DIR}/"
)

print()

print("3. TFLite:")
print(
    f"   {TFLITE_PATH}"
)

print()

print("4. Vocabulary:")
print(
    f"   {VOCAB_PATH}"
)

print()

print("5. Config:")
print(
    "   model_config.json"
)

print()

print("=" * 60)