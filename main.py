"""
Thai Next Word Prediction - Tkinter desktop app.

Loads `thai_next_word_transformer.tflite` and `thai_vocab.json`, and lets the
user type Thai text in a textbox. As they type, the app suggests the next
word using the transformer model and lets them insert a suggestion with a
click (or Tab for the top suggestion).
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import tensorflow as tf
from pythainlp.tokenize import word_tokenize

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "thai_next_word_transformer.tflite")
VOCAB_PATH = os.path.join(BASE_DIR, "thai_vocab.json")

SEQ_LENGTH = 100
TOP_K = 5

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, START_TOKEN}


# ============================================================
# MODEL / VOCAB WRAPPER
# ============================================================

class ThaiNextWordPredictor:
    """Wraps the TFLite transformer model and vocabulary lookup."""

    def __init__(self, model_path, vocab_path, seq_length=SEQ_LENGTH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        self.seq_length = seq_length

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.word_to_id = json.load(f)

        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

        self.pad_id = self.word_to_id.get(PAD_TOKEN, 0)
        self.unk_id = self.word_to_id.get(UNK_TOKEN, 1)
        self.start_id = self.word_to_id.get(START_TOKEN, 2)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.runner = self.interpreter.get_signature_runner("serving_default")

        input_details = self.interpreter.get_input_details()[0]
        self._input_key = self.interpreter.get_signature_list()["serving_default"][
            "inputs"
        ][0]
        self._output_key = self.interpreter.get_signature_list()["serving_default"][
            "outputs"
        ][0]

        # Respect the model's expected sequence length if different.
        model_seq_len = input_details["shape"][-1]
        if model_seq_len and model_seq_len > 0:
            self.seq_length = int(model_seq_len)

    def tokenize(self, text):
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        return [t for t in tokens if t.strip()]

    def _encode(self, tokens):
        ids = [self.start_id] + [
            self.word_to_id.get(tok, self.unk_id) for tok in tokens
        ]

        if len(ids) < self.seq_length:
            ids = [self.pad_id] * (self.seq_length - len(ids)) + ids
        else:
            ids = ids[-self.seq_length :]

        return np.array([ids], dtype=np.int32)

    def predict_next_words(self, text, top_k=TOP_K):
        """Return a list of (word, probability) sorted by probability desc."""

        tokens = self.tokenize(text)

        if not tokens:
            return []

        input_ids = self._encode(tokens)

        outputs = self.runner(**{self._input_key: input_ids})
        probs = outputs[self._output_key][0]

        top_indices = np.argsort(probs)[::-1]

        suggestions = []

        for idx in top_indices:
            word = self.id_to_word.get(int(idx))

            if word is None or word in SPECIAL_TOKENS:
                continue

            suggestions.append((word, float(probs[idx])))

            if len(suggestions) >= top_k:
                break

        return suggestions


# ============================================================
# TKINTER APP
# ============================================================

class NextWordApp:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor

        root.title("Thai Next Word Prediction")
        root.geometry("640x420")
        root.minsize(480, 360)

        self._build_widgets()
        self._pending_after_id = None

    def _build_widgets(self):
        padding = {"padx": 12, "pady": 8}

        title_label = ttk.Label(
            self.root,
            text="พิมพ์ข้อความภาษาไทย แล้วดูคำแนะนำคำต่อไป",
            font=("TH Sarabun New", 14, "bold"),
        )
        title_label.pack(anchor="w", **padding)

        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill="both", expand=True, padx=12)

        self.text_box = tk.Text(
            text_frame,
            height=10,
            wrap="word",
            font=("TH Sarabun New", 16),
            undo=True,
        )
        self.text_box.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.text_box.yview
        )
        scrollbar.pack(fill="y", side="right")
        self.text_box.configure(yscrollcommand=scrollbar.set)

        self.text_box.bind("<KeyRelease>", self._on_key_release)
        self.text_box.bind("<Tab>", self._on_tab_pressed)

        suggestion_label = ttk.Label(
            self.root, text="คำแนะนำ (คลิกเพื่อแทรก):", font=("TH Sarabun New", 11)
        )
        suggestion_label.pack(anchor="w", padx=12, pady=(8, 0))

        self.suggestion_frame = ttk.Frame(self.root)
        self.suggestion_frame.pack(fill="x", padx=12, pady=(4, 12))

        self.suggestion_buttons = []
        for _ in range(TOP_K):
            btn = ttk.Button(self.suggestion_frame, text="", state="disabled")
            btn.pack(side="left", padx=4, fill="x", expand=True)
            self.suggestion_buttons.append(btn)

        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", padx=12, pady=(0, 10))

        self.status_label = ttk.Label(
            status_bar, text="พร้อมใช้งาน", foreground="gray"
        )
        self.status_label.pack(side="left")

        clear_btn = ttk.Button(status_bar, text="ล้างข้อความ", command=self._clear_text)
        clear_btn.pack(side="right")

    # --------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------

    def _on_key_release(self, event):
        # Ignore pure navigation / modifier keys to avoid redundant predicts.
        if event.keysym in (
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Alt_L",
            "Alt_R",
            "Left",
            "Right",
            "Up",
            "Down",
        ):
            return

        if self._pending_after_id is not None:
            self.root.after_cancel(self._pending_after_id)

        # Debounce: predicting on every keystroke is wasteful since the
        # transformer forward pass takes a noticeable moment.
        self._pending_after_id = self.root.after(250, self._update_suggestions)

    def _on_tab_pressed(self, event):
        if self.suggestion_buttons and self.suggestion_buttons[0]["state"] != "disabled":
            self._insert_word(self.suggestion_buttons[0]["text"])
            return "break"  # prevent default Tab focus change
        return None

    def _update_suggestions(self):
        text = self.text_box.get("1.0", "end-1c")

        self.status_label.configure(text="กำลังทำนาย...")
        self.root.update_idletasks()

        try:
            suggestions = self.predictor.predict_next_words(text, top_k=TOP_K)
        except Exception as exc:  # surfaces model/runtime errors in the UI
            self.status_label.configure(text=f"เกิดข้อผิดพลาด: {exc}")
            return

        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                word, prob = suggestions[i]
                btn.configure(
                    text=f"{word} ({prob * 100:.1f}%)",
                    state="normal",
                    command=lambda w=word: self._insert_word(w),
                )
            else:
                btn.configure(text="", state="disabled", command=lambda: None)

        self.status_label.configure(text="พร้อมใช้งาน")

    def _insert_word(self, label_text):
        # Button label includes the probability; extract the raw word.
        word = label_text.split(" (")[0]

        if not word:
            return

        current = self.text_box.get("1.0", "end-1c")

        if current and not current.endswith((" ", "\n")):
            self.text_box.insert("end", " ")

        self.text_box.insert("end", word + " ")
        self.text_box.see("end")
        self._update_suggestions()

    def _clear_text(self):
        self.text_box.delete("1.0", "end")
        for btn in self.suggestion_buttons:
            btn.configure(text="", state="disabled", command=lambda: None)
        self.status_label.configure(text="พร้อมใช้งาน")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    try:
        predictor = ThaiNextWordPredictor(MODEL_PATH, VOCAB_PATH)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("โหลดโมเดลล้มเหลว", str(exc))
        root.destroy()
        return

    root = tk.Tk()
    NextWordApp(root, predictor)
    root.mainloop()


if __name__ == "__main__":
    main()
