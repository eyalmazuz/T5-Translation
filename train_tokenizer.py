import argparse

import sentencepiece as spm
from transformers import T5TokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, nargs='+', required=True, help='Path to corpora files')
    parser.add_argument('--model_prefix', type=str, default='m', help='Name of the generated tokenizer file')
    parser.add_argument('--save_path', type=str, required=True, default='./', help='Where to save the T5 tokenizer')
    parser.add_argument('--vocab_size', type=int, default=35000, help='Number of tokens to learn')
    parser.add_argument('--input_sentence_size', type=int, default=0, help='Number of sentences to use')

    return parser.parse_args()


def main():
    args = parse_args()

    spm.SentencePieceTrainer.train(input=','.join(args.data), model_prefix=args.model_prefix,
                                   vocab_size=args.vocab_size, pad_id=0, unk_id=1,
                                   bos_id=2, eos_id=3, pad_piece='<pad>', unk_piece='<unk>',
                                   bos_piece='<s>', eos_piece='</s>', model_type='unigram',
                                   input_sentence_size=args.input_sentence_size,
                                   shuffle_input_sentence=True)

    tokenizer = T5TokenizerFast(f'{args.model_prefix}.model', bos_token='<s>')
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__": 
    main()
