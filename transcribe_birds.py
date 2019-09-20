import argparse
import sys
import warnings

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import json
import tqdm

sys.path.append(os.getcwd())
from data_preprocess.utils import read_class_name, read_images_filename, read_images_class_label, classify_images_by_class, check_dir, read_class_ids

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model-path', default='./deepspeech.pytorch/model/librispeech_pretrained.pth',
                    help='Path to model file created by training')
parser.add_argument('--audio-path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
parser.add_argument('--output_filename', type=str, default="text_asr.json", help="output json filename")
parser.add_argument('--dataset', choices=['birds', 'flowers'], default='birds', help="")

beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm-path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff-top-n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm-workers', default=4, type=int, help='Number of LM processes to use')

args = parser.parse_args()


def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }
    results['_meta']['acoustic_model'].update(DeepSpeech.get_meta(model))

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def process_a_file(model, decoder, filename):
    spect = parser.parse_audio(filename).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(spect.requires_grad_(False))
    decoded_output, decoded_offsets = decoder.decode(out.data)
    json_data = decode_results(model, decoded_output, decoded_offsets)
    return json_data['output'][0]['transcription']


def process_all_file(model, decoder):
    root = os.getcwd()
    audio_data_dir = os.path.join(root, "./data/birds/CUB_200_2011_audio/", "audio")
    audio_asr_data_dir = os.path.join(root, "./data/birds/CUB_200_2011_audio_asr/", "audio")

    class_name_filename, test_ids_filename, class_label_filename, images_filename = \
            os.path.join(root, "data", "./birds/CUB_200_2011/classes.txt"), \
            os.path.join(root, 'data', "./birds/CUB_200_2011/valids.txt"), \
            os.path.join(root, "data", "./birds/CUB_200_2011/image_class_labels.txt"), \
            os.path.join(root, "data", "./birds/CUB_200_2011/images.txt")

    class_name_all, test_ids, images_label_all, images_filename_all = \
            read_class_name(class_name_filename),\
            read_class_ids(test_ids_filename), \
            read_images_class_label(class_label_filename),\
            read_images_filename(images_filename)
    images_filename_grouped_by_class = classify_images_by_class(class_name_all, images_filename_all, images_label_all)

    # save to a json file
    
    # person_id = ['0', '1', '3', '4']
    person_id = ['0']
    check_dir(audio_asr_data_dir)
    for p_id in person_id:
        print("process audio data with id ({})".format(p_id))
        check_dir(os.path.join(audio_asr_data_dir, p_id))
        asr_data_all = {}
        bar = tqdm.tqdm(test_ids)
        for class_id in bar:
            class_name = class_name_all[class_id]
            asr_data_all[class_name] = {}
            for image_filename in images_filename_grouped_by_class[class_id]:
                filename = os.path.splitext(image_filename)[0]
                asr_data_all[class_name][filename] = []
                bar.set_description("{}:{}".format(class_name, filename))
                for idx in range(10):
                    audio_filename_all = os.path.join(audio_data_dir, p_id, filename+"_{}.wav".format(str(idx)))
                    assert(os.path.isfile(audio_filename_all))
                    text_data = process_a_file(model, decoder, audio_filename_all).lower()
                    asr_data_all[class_name][filename].append(text_data)
            # write to audio_asr_data
            asr_filename = os.path.join(audio_asr_data_dir, p_id, args.output_filename)
            with open(asr_filename, 'w') as fp:
                json.dump(asr_data_all, fp, indent=4, sort_keys=True)
        print("process audio data with id ({}) end, save to file: {}".format(p_id, asr_filename))


import copy
def process_birds(model, decoder):
    root = os.getcwd()
    data_folder = os.path.join(root, "./data/flowers")
    audio_folder = os.path.join(data_folder, 'audio')
    for split in ('train', 'test'):
        print("process {} data".format(split))
        with open(os.path.join(data_folder, "{}.json".format(split)), 'r') as fp:
            json_data_all = json.load(fp)
        json_data_all_new = copy.deepcopy(json_data_all)
        bar = tqdm.tqdm(json_data_all['data'])
        for idx, _d in enumerate(bar):
            audio_paths = _d['audio']
            text_datas = [process_a_file(model, decoder, os.path.join(audio_folder, audio_path)) for audio_path in audio_paths]
            json_data_all_new['data'][idx]['asr'] = text_datas
        with open(os.path.join(data_folder, "{}_asr.json".format(split)), 'w') as fp:
            json.dump(json_data_all_new, fp, indent=4, sort_keys=True)

def process_flowers(model, decoder):
    root = os.getcwd()
    data_folder = os.path.join(root, "./data/flowers")
    audio_folder = os.path.join(data_folder, 'audio')
    for split in ('train', 'test'):
        print("process {} data".format(split))
        with open(os.path.join(data_folder, "{}.json".format(split)), 'r') as fp:
            json_data_all = json.load(fp)
        json_data_all_new = copy.deepcopy(json_data_all)
        bar = tqdm.tqdm(json_data_all['data'])
        for idx, _d in enumerate(bar):
            audio_paths = _d['wav']
            text_datas = [process_a_file(model, decoder, os.path.join(audio_folder, audio_path)) for audio_path in audio_paths]
            json_data_all_new['data'][idx]['asr'] = text_datas
        with open(os.path.join(data_folder, "{}_asr.json".format(split)), 'w') as fp:
            json.dump(json_data_all_new, fp, indent=4, sort_keys=True)
            



if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    if args.dataset == 'birds':
        process_birds(model, decoder)
    elif args.dataset == 'flowers':
        process_flowers(model, decoder)
    else:
        raise NotImplementedError
    # process_all_file(model, decoder)
    

    # spect = parser.parse_audio(args.audio_path).contiguous()
    # spect = spect.view(1, 1, spect.size(0), spect.size(1))
    # out = model(Variable(spect, volatile=True))
    # decoded_output, decoded_offsets = decoder.decode(out.data)
    # print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
