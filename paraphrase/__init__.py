from .bart import bart_paraphrase
from .t5 import t5_paraphrase

PARAPHRASERS = {
    'bart': bart_paraphrase,
    't5': t5_paraphrase,
}
