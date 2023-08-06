import soundfile
from onnx_inference import OnnxRVC
import wave
import os
import traceback, pdb
from fairseq import checkpoint_utils
from config import Config
import numpy as np
import scipy.io.wavfile as wavfile
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio, CSVutil
import torch
import re as regex
from vc_infer_pipeline import VC

global DoFormant, Quefrency, Timbre
try:
    DoFormant, Quefrency, Timbre = CSVutil("csvdb/formanting.csv", "r", "formanting")
    DoFormant = (
        lambda DoFormant: True
        if DoFormant.lower() == "true"
        else (False if DoFormant.lower() == "false" else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil("csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre)


config = Config()

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "./logs/"
audio_root = "audios"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []

global indexes_list
indexes_list = []

audio_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s\\%s" % (root, name))

for root, dirs, files in os.walk(audio_root, topdown=False):
    for name in files:
        audio_paths.append("%s/%s" % (root, name))

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


# Only one timbre can be set globally for a tab
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # Considering polling, it's necessary to add a condition to check whether the 'sid' has switched from having a model to not having a model
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ### If you don't go through this process downstairs, the cleanup won't be thorough
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return (
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


def vc_single(
    sid,
    input_audio_path0,
    input_audio_path1,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path0 is None or input_audio_path0 is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        if input_audio_path0 == "":
            audio = load_audio(input_audio_path1, 16000, DoFormant, Quefrency, Timbre)

        else:
            audio = load_audio(input_audio_path0, 16000, DoFormant, Quefrency, Timbre)

        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path1,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def get_wav_sampling_rate(file_path):
    wav_file = wave.open(file_path, "rb")
    sampling_rate = wav_file.getframerate()
    wav_file.close()
    return sampling_rate

def cli_split_command(com):
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = regex.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array


def cli_infer(com):
    # get VC first
    com = cli_split_command(com)
    model_name = com[0]
    source_audio_path = com[1]
    output_file_name = com[2]
    feature_index_path = com[3]
    f0_file = None  # Not Implemented Yet

    # Get parameters for inference
    speaker_id = int(com[4])
    transposition = float(com[5])
    f0_method = com[6]
    crepe_hop_length = int(com[7])
    harvest_median_filter = int(com[8])
    resample = int(com[9])
    mix = float(com[10])
    feature_ratio = float(com[11])
    protection_amnt = float(com[12])
    protect1 = 0.5

    if com[13] == "False" or com[13] == "false":
        DoFormant = False
        Quefrency = 0.0
        Timbre = 0.0
        CSVutil(
            "csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
        )

    else:
        DoFormant = True
        Quefrency = float(com[14])
        Timbre = float(com[15])
        CSVutil(
            "csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
        )

    print("Mangio-RVC-Fork Infer-CLI: Starting the inference...")
    vc_data = get_vc(model_name, protection_amnt, protect1)
    print(vc_data)
    print("Mangio-RVC-Fork Infer-CLI: Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        source_audio_path,
        transposition,
        f0_file,
        f0_method,
        feature_index_path,
        feature_index_path,
        feature_ratio,
        harvest_median_filter,
        resample,
        mix,
        protection_amnt,
        crepe_hop_length,
    )
    if "Success." in conversion_data[0]:
        print(
            "Mangio-RVC-Fork Infer-CLI: Inference succeeded. Writing to %s/%s..."
            % ("audio-outputs", output_file_name)
        )
        wavfile.write(
            "%s/%s" % ("audio-outputs", output_file_name),
            conversion_data[1][0],
            conversion_data[1][1],
        )
        print(
            "Mangio-RVC-Fork Infer-CLI: Finished! Saved output to %s/%s"
            % ("audio-outputs", output_file_name)
        )
    else:
        print("Mangio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ")
        print(conversion_data[0])

cli_infer("Spongebob.pth audios/out.wav audios/spongebob_out.wav logs/added_IVF6717_Flat_nprobe_1_Spongebob_v2.index 0 8 harvest 120 3 0 1 0.78 0.33 false")

# hop_size = 120
# # sampling_rate = 40000  # Sample rate
# f0_up_key = 0  # Pitch shift
# sid = 0  # Role ID
# f0_method = "harvest"  # F0 extraction algorithm
# model_path = "weights/Spongebob.onnx"  # Full path to the model
# vec_name = "weights/vec-768-layer-12.onnx"  # Automatically completed to f"pretrained/{vec_name}.onnx" for the vec model in ONNX format
# wav_path = "Python-Wrapper-for-World-Vocoder/demo/utterance/vaiueo2d.wav"  # Input path or ByteIO instance for the WAV file
# # wav_path = "out.wav"
# out_path = "out_spongebob_onnx.wav"  # O utput path or ByteIO instance for the processed WAV file
# sampling_rate = get_wav_sampling_rate(wav_path)
# model = OnnxRVC(
#     model_path, vec_path=vec_name, sr=sampling_rate, hop_size=hop_size, device="cuda"
# )
# audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)
# soundfile.write(out_path, audio, sampling_rate)
