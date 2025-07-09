import torch
import h5py
import re
import torch.nn.functional as F

# ---------- 1. pre-compiled regexes ----------
# (a) Full math environments: $$ … $$, $ … $, \( … \), \[ … \]
_MATH_BLOCKS = re.compile(
    r"""
      (?s)                 # DOTALL – make '.' match newlines
      \$\$.+?\$\$          |      # $$ … $$
      \$.+?\$              |      # $ … $
      \\\(.+?\\\)          |      # \( … \)
      \\\[.+?\\\]                 # \[ … \]
    """,
    re.VERBOSE,
)

# (b) Control sequences:  \command   \command{…}   \command[opt]{…}
#    Keeps the *arguments* (text inside braces/brackets) but drops the backslash+name.
_LATEX_CMDS = re.compile(r"\\[A-Za-z]+[*]?")  # star form allowed (e.g., \section*)

# (c) Remove any remaining bare braces / brackets
_BRACES = re.compile(r"[{}\[\]]")

# ---------- 2. regex for Japanese characters ----------
#_JP_CHARS = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u4E00-\u9FFF]")
_JP_CHARS = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F\uFF00-\uFFEF]')

# ---------- 3. main helper ----------
def count_japanese_no_latex(text: str) -> int:
    """Return the number of Japanese characters in *text*
    after discarding LaTeX markup."""
    # Strip math blocks entirely
    cleaned = _MATH_BLOCKS.sub("", text)

    # Drop the control words (‘\command’) but leave their arguments intact
    cleaned = _LATEX_CMDS.sub("", cleaned)

    # Eliminate leftover braces / brackets
    cleaned = _BRACES.sub("", cleaned)

    # Finally count Japanese characters
    return len(_JP_CHARS.findall(cleaned)), len(cleaned)

def clean_latex(text: str) -> str:
    """Return the number of Japanese characters in *text*
    after discarding LaTeX markup."""
    # Strip math blocks entirely
    cleaned = _MATH_BLOCKS.sub("", text)

    # Drop the control words (‘\command’) but leave their arguments intact
    cleaned = _LATEX_CMDS.sub("", cleaned)

    # Eliminate leftover braces / brackets
    cleaned = _BRACES.sub("", cleaned)
    return cleaned


def load_h5py(filename):
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            data[key] = torch.from_numpy(f[key][()])
    return data

def save_h5py(dict, filename):
    with h5py.File(filename, 'w') as f:
        for key, value in dict.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            f.create_dataset(key, data=value, compression='gzip')

def reduce_padding_general(data, pad_value=-100):
    # data: (N, M)
    mask = (data != pad_value)
    
    # Find last non-padded token per sequence
    idx = mask.float().cumsum(dim=1).argmax(dim=1)
    
    # Determine maximum length across batch
    max_len = (idx + 1).max().item()
    
    # Slice the data up to max_len
    reduced_data = data[:, :max_len]

    return reduced_data

def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers

if __name__ == '__main__':
    path = 'ckpts/generated_data_0_Qwen2.5-3B-Instruct_logit.h5'
    data = load_h5py(path)
