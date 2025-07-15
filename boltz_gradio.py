import os, io, re, uuid, json, time, torch, base64
import shutil, zipfile, requests, tempfile, subprocess, threading, contextlib
import numpy as np
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from yaml import safe_dump, safe_load
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity
from rdkit.Contrib.SA_Score import sascorer # type: ignore
from rdkit.Contrib.NP_Score import npscorer # type: ignore
from pathlib import Path

from boltz.main import download_boltz2
from boltz.data import const

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from functools import partial

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.cealign import CEAligner
from Bio.PDB.mmcifio import MMCIFIO

from posebusters import PoseBusters

# TODO: Convert AF3/Chai-1/Protenix JSON to Boltz YAML

RDLogger.DisableLog('rdApp.*')
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    fscore = npscorer.readNPModel()

periodic_table = Chem.GetPeriodicTable()

entity_types = ['Protein', 'DNA', 'RNA', 'Ligand', 'CCD']
entity_label_map = {'Protein': 'Sequence', 'DNA': 'Sequence', 'RNA': 'Sequence',
                    'Ligand': 'SMILES', 'CCD': 'CCD Code'}

allow_char_dict = {'Protein': "ACDEFGHIKLMNPQRSTVWY",
                   'DNA'    : "ACGT",
                   'RNA'    : "ACGU"}

rev_comp_map = {'DNA': {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'U': 'A'},
                'RNA': {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C', 'T': 'A'}}

property_functions = {'Molecular Weight'  : Descriptors.MolWt,
                      'Num. of Hydrogen Bond Donors' : Descriptors.NumHDonors,
                      'Num. of Hydrogen Bond Acceptors' : Descriptors.NumHAcceptors,
                      'LogP': Descriptors.MolLogP,
                      'Topological Polar Surface Area (TPSA)': Descriptors.TPSA,
                      'Rotatable Bonds'  : Descriptors.NumRotatableBonds,
                      'Num. of Rings' : Descriptors.RingCount,
                      'Formal Charge'  : lambda mol: sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
                      'Num. of Heavy Atoms' : Descriptors.HeavyAtomCount,
                      'Num. of Atoms'  : lambda mol: mol.GetNumAtoms(),
                      'Molar Refractivity'  : Descriptors.MolMR,
                      'Quantitative Estimate of Drug-Likeness (QED)' : Descriptors.qed,
                      'Natural Product-likeness Score (NP)': partial(npscorer.scoreMol, fscore=fscore),
                      'Synthetic Accessibility Score (SA)': sascorer.calculateScore}

file_extract_matching_map = {'Structure' : ['.cif', '.sdf', '_bust.csv'],
                             'Confidence': ['confidence_'],
                             'Affinity'  : ['affinity_'],
                             'PAE'       : ['pae_'],
                             'PDE'       : ['pde_'],
                             'pLDDT'     : ['plddt_']}

css = """
footer { display: none !important; }
.sequence textarea {font-family: Courier New, Courier, monospace; !important}
.validation {font-size: 12px; font-family: Courier New, Courier, monospace; !important}
.log textarea {font-size: 12px; font-family: Courier New, Courier, monospace; !important}
.small-upload-style .wrap {font-size: 10px; !important}
.small-upload-style .icon-wrap svg {display: none; !important}
.small-header-table th {font-size: 10px !important; text-align: center !important;}
.centered-checkbox {display: flex; align-items: center; padding: 0 10px;}
"""

device_num = 1
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_num = torch.cuda.device_count()

curr_dir = os.path.dirname(__file__)
output_dir = os.path.join(curr_dir, 'boltz_output')
# output_dir = os.path.join(curr_dir, 'boltz_vhts')
os.makedirs(output_dir, exist_ok=True)

template_dir = os.path.join(curr_dir, 'templates')
os.makedirs(template_dir, exist_ok=True)

msa_dir = os.path.join(curr_dir, 'usr_msa')
os.makedirs(msa_dir, exist_ok=True)

input_dir = os.path.join(curr_dir, 'boltz_input')
os.makedirs(input_dir, exist_ok=True)
shutil.rmtree(input_dir)
os.makedirs(input_dir)

### Defining Boltz parameters ###
device_number = gr.Number(device_num, label='devices',
                          info='The number of devices to use for prediction.',
                          minimum=1, maximum=device_num, interactive=True)
acc_choices = ['gpu', 'cpu'] if cuda_available else ['cpu']
accelerator_type = gr.Dropdown(acc_choices, value=acc_choices[0],
                                info='The accelerator to use for prediction.',
                                label='accelerator', interactive=True)
recycling_steps = gr.Number(3, label='recycling_steps',
                            info='The number of recycling steps to use for prediction.',
                            minimum=1, interactive=True)
sampling_steps  = gr.Number(200, label='sampling_steps',
                            info='The number of sampling steps to use for prediction.',
                            minimum=1, interactive=True)
diffusion_samples = gr.Number(3, label='diffusion_samples', 
                              info='The number of diffusion samples to use for prediction.',
                              minimum=1, interactive=True)
step_scale = gr.Number(1.638, label='step_scale',
                       info='The lower the higher the diversity among samples (recommended between 1 and 2).',
                       minimum=0., interactive=True)
num_workers = gr.Number(2, label='num_workers', minimum=0,
                        info='The number of dataloader workers to use for prediction.',
                        maximum=os.cpu_count(), interactive=True)
preprocessing_threads = gr.Number(os.cpu_count(), label='preprocessing-threads',
                                  info='The number of threads to use for preprocessing.',
                                  interactive=True, minimum=1, maximum=os.cpu_count())
affinity_mw_correction = gr.Checkbox(False, label='affinity_mw_correction',
                                     info='Whether to add the Molecular Weight correction to the affinity value head.',
                                     interactive=True,)
sampling_steps_affinity = gr.Number(200, label='sampling_steps_affinity',
                                    info='The number of sampling steps to use for affinity prediction.',
                                    interactive=True, minimum=1)
diffusion_samples_affinity = gr.Number(5, label='diffusion_samples_affinity',
                                       info='The number of diffusion samples to use for affinity prediction.',
                                       interactive=True, minimum=1)
no_kernels = gr.Checkbox(False if cuda_available else True, label='no_kernels',
                         info='Whether to NOT use trifast kernels for triangular updates.')
override = gr.Checkbox(False, label='override', info='Whether to override existing predictions if found.')
use_potentials = gr.Checkbox(False, label='use_potentials',
                             info='Whether to run the original Boltz-2 model using inference time potentials.')
boltz_method = gr.Dropdown(list(const.method_types_ids.keys()), label='method',
                           value='x-ray diffraction',
                           info='The method to use for prediction.')

all_boltz_parameters = [device_number, accelerator_type, recycling_steps, sampling_steps,
                        diffusion_samples, step_scale, num_workers, preprocessing_threads,
                        affinity_mw_correction, sampling_steps_affinity, diffusion_samples_affinity,
                        use_potentials, boltz_method, no_kernels, override]

def concurrent_download_model_weight():
    cache_pth = Path('~/.boltz').expanduser()
    cache_pth.mkdir(exist_ok=True)
    all_files = os.listdir(cache_pth)
    if ('mols' in all_files and 'ccd.pkl' in all_files and 
        'boltz2_conf.ckpt' in all_files and 'boltz2_aff.ckpt' in all_files):
        return
    download_boltz2(cache_pth)
    return

def manual_download_boltz_weights():
    cache_pth = Path('~/.boltz').expanduser()
    cache_pth.mkdir(exist_ok=True)
    all_files = os.listdir(cache_pth)
    if ('mols' in all_files and 'ccd.pkl' in all_files and 
        'boltz2_conf.ckpt' in all_files and 'boltz2_aff.ckpt' in all_files):
        yield gr.update(interactive=True, value='Weight downloaded!')
    yield gr.update(interactive=False, value='Downloading...')
    download_boltz2(cache_pth)
    yield gr.update(interactive=True, value='Weight downloaded!')

### Boltz parameters end ###

def check_dir_exist_and_rename(dir_pth: str):
    basename = os.path.basename(dir_pth).rsplit('_', 1)[0]
    dirname = os.path.dirname(dir_pth)
    while os.path.isdir(dir_pth):
        dir_pth = os.path.join(dirname, f'{basename}_{uuid.uuid4().hex[:8]}')
    os.makedirs(dir_pth)

def _check_yaml_strings(yaml_str: str):
    if not yaml_str:
        return False
    yaml_dict = safe_load(yaml_str)
    if 'sequences' not in yaml_dict or len(yaml_dict['sequences']) < 1:
        return False
    for seq_dict in yaml_dict['sequences']:
        k = list(seq_dict.keys())[0]
        if k not in ['protein', 'ligand', 'rna', 'dna'] or len(seq_dict) > 1:
            return False
        seq_info_dict = seq_dict[k]
        if 'id' not in seq_info_dict or ('sequence' not in seq_info_dict and 
                                         'smiles'   not in seq_info_dict and
                                         'ccd'      not in seq_info_dict):
            return False
    return True

def check_yaml_strings(yaml_str: str, *args):
    final_bool_args = []
    for value in args:
        if isinstance(value, pd.DataFrame):
            final_bool_args.append(not value.empty)
        else:
            final_bool_args.append(value)
    return gr.update(interactive=_check_yaml_strings(yaml_str) & all(final_bool_args))

def check_batch_yaml_and_name(yaml_str: str, name_str: str):
    name_valid = bool(name_str.strip())
    yaml_valid = _check_yaml_strings(yaml_str)
    validity_text = ''
    if not name_valid:
        validity_text += 'Missing name. '
    if not yaml_valid:
        validity_text += 'Invalid yaml file.'
    return gr.update(info=validity_text)

def clear_curr_batch_dict():
    return {}, 0

def upload_multi_files(files: list[str], curr_cnt: int):
    final_yaml_dict = {}
    for file in files:
        base_name = os.path.basename(file).rsplit('.', 1)[0]
        with open(file) as f:
            yaml_str = f.read()
            if _check_yaml_strings(yaml_str):
                final_yaml_dict[base_name] = yaml_str
        os.remove(file)
    curr_cnt += len(final_yaml_dict)
    return final_yaml_dict, curr_cnt, None

def add_current_single_to_batch(name: str, yaml_str: str, curr_yaml_dict: dict, curr_cnt: int):
    if name in curr_yaml_dict:
        i = 2
        new_name = f'{name}_{i}'
        while new_name in curr_yaml_dict:
            i += 1
            new_name = f'{name}_{i}'
        name = new_name
    curr_yaml_dict[name] = yaml_str
    yield curr_yaml_dict, curr_cnt + 1, 'Complex added!'
    time.sleep(2.)
    yield gr.update(), gr.update(), 'Add to Batch'

def read_tempaltes(files: list[str], old_cif_name_chain_dict: dict,
                   old_cif_name_path_dict: dict, old_usage_dict: dict,
                   old_template_name_setting_dict: dict):
    if not old_cif_name_path_dict:
        saved_cif_dir = os.path.join(template_dir, uuid.uuid4().hex[:8])
        check_dir_exist_and_rename(saved_cif_dir)
    else:
        written_file = list(old_cif_name_path_dict.values())[0]
        saved_cif_dir = os.path.dirname(written_file)
    for cif_file in files:
        name = os.path.basename(cif_file).rsplit('.', 1)[0]
        new_template_pth = os.path.join(saved_cif_dir, os.path.basename(cif_file))
        chain_index = 0
        stop_search_chain = False
        unique_chains = set()
        with open(cif_file) as f:
            cif_str = f.read()
        with open(new_template_pth, 'w') as f:
            f.write(cif_str)
        for line in cif_str.splitlines():
            if line.startswith('_atom_site.') and not stop_search_chain:
                label = line.strip().split('_atom_site.', 1)[-1]
                if label == 'label_asym_id':
                    stop_search_chain = True
                else:
                    chain_index += 1
            elif line.startswith(('HETATM', 'ATOM')):
                chain = line.split()[chain_index]
                unique_chains.update(chain)
            elif stop_search_chain and line.strip() == '#':
                break
        old_cif_name_chain_dict[name] = sorted(list(unique_chains))
        old_cif_name_path_dict[name] = new_template_pth
        old_template_name_setting_dict[name] = {'chain_id': [], 'template_id': [], 'force': False, 'threshold': -1}
        if name not in old_usage_dict:
            old_usage_dict[name] = True
    return (gr.update(choices=list(old_cif_name_chain_dict), value=list(old_cif_name_chain_dict)[0]),
            old_cif_name_chain_dict, old_cif_name_path_dict, old_usage_dict,
            gr.update(interactive=bool(old_usage_dict), value=old_usage_dict[list(old_cif_name_chain_dict)[0]]),
            old_template_name_setting_dict, gr.update(value=False, interactive=bool(old_usage_dict)),
            gr.update(value=-1, interactive=bool(old_usage_dict)))

def update_template_chain_ids_and_settings(curr_usage_bool: bool, target_chain_ids: list, template_chain_ids: list,
                                           force_template: bool, template_threshold: int,
                                           curr_name: str,
                                           template_name_usage_dict: dict, template_name_setting_dict: dict):
    template_name_usage_dict[curr_name] = curr_usage_bool
    template_name_setting_dict[curr_name]['chain_id'] = target_chain_ids
    template_name_setting_dict[curr_name]['template_id'] = template_chain_ids
    template_name_setting_dict[curr_name]['force'] = force_template
    template_name_setting_dict[curr_name]['threshold'] = template_threshold
    return template_name_usage_dict, template_name_setting_dict

def update_template_dropdown(curr_name: str, template_name_chain_dict: dict,
                             template_name_usage_dict: dict, template_name_setting_dict: dict):
    return (template_name_usage_dict[curr_name],
            template_name_setting_dict[curr_name]['chain_id'],
            gr.update(value=template_name_setting_dict[curr_name]['template_id'],
                      choices=template_name_chain_dict[curr_name]),
            gr.update(value=template_name_setting_dict[curr_name]['force']),
            gr.update(value=template_name_setting_dict[curr_name]['threshold']))

def update_bond_sequence_length_with_chain(sel_chain: str, mapping_dict: dict):
    data_dict = mapping_dict.get(sel_chain, None)
    if data_dict is None:
        return gr.update(choices=None, value=None)
    if data_dict['type'] in ['CCD']:
        return gr.update(choices=['1'], value='1', interactive=True)
    elif data_dict['type'] in ['Protein', 'DNA', 'RNA']:
        total_len = len(data_dict['sequence'])
        return gr.update(choices=[str(i) for i in range(1, total_len+1)], value='1', interactive=True)
    else:
        return gr.update(choices=None, value=None)

def _combine_cif_models_to_dict(target_files: list):
    mmcif_parser = MMCIFParser()
    ref_cif = target_files[0]
    ref_struct = mmcif_parser.get_structure('ref', ref_cif)
    
    aligner = CEAligner()
    aligner.set_reference(ref_struct)
    
    combined_dict = MMCIF2Dict(target_files[0])
    atom_site_keys = [k for k in combined_dict.keys() if k.startswith('_atom_site.')]
    
    mmcif_io = MMCIFIO()
    
    for i, cif_file in enumerate(target_files[1:]):
        s = mmcif_parser.get_structure('target', cif_file)
        t_dict = MMCIF2Dict(cif_file)
        
        aligner.align(s)
        
        mmcif_io.set_structure(s)
        cif_io = io.StringIO()
        mmcif_io.save(cif_io)
        cif_io.seek(0)
        
        a_dict = MMCIF2Dict(cif_io)
        for k in ['_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z']:
            t_dict[k] = a_dict[k]
        
        for k in atom_site_keys:
            if k != '_atom_site.pdbx_PDB_model_num':
                combined_dict[k].extend(t_dict[k])
            else:
                model_num = [f'{i+2}'] * len(t_dict[k])
                combined_dict[k].extend(model_num)
    
    return combined_dict

def _write_cif_dict_to_cif(cif_dict: dict, out_pth: str):
    cif_io = MMCIFIO()
    cif_io.set_dict(cif_dict)
    cif_io.save(str(out_pth))

def combine_and_write_cif(target_files: list, out_pth: str|Path):
    _write_cif_dict_to_cif(_combine_cif_models_to_dict(target_files), out_pth)

### Running Boltz ###
def execute_single_boltz(file_name: str, yaml_str: str,
                         devices: int, accelerator: str,
                         recycling_steps: int, sampling_steps: int,
                         diffusion_samples: float, step_scale: int,
                         num_workers: int, preprocessing_threads: int,
                         affinity_mw_correction: bool,
                         sampling_steps_affinity: int, diffusion_samples_affinity: int,
                         use_potentials: bool, boltz_method: str, no_kernels: bool, override: bool):
    random_dir_name = f"{file_name}_{uuid.uuid4().hex[:8]}"
    inp_rng_dir = os.path.join(input_dir, random_dir_name)
    out_rng_dir = os.path.join(output_dir, random_dir_name)
    check_dir_exist_and_rename(inp_rng_dir)
    check_dir_exist_and_rename(out_rng_dir)
    inp_yaml = os.path.join(inp_rng_dir, file_name+'.yaml')
    with open(inp_yaml, 'w') as f:
        f.write(yaml_str)
    final_strs = ['--use_msa_server', '--write_full_pae', '--write_full_pde']
    if use_potentials:
        final_strs.append('--use_potentials')
    if affinity_mw_correction:
        final_strs.append('--affinity_mw_correction')
    if no_kernels:
        final_strs.append('--no_kernels')
    if override:
        final_strs.append('--override')
    cmd = ['boltz', 'predict', inp_yaml,
           '--out_dir', out_rng_dir,
           '--devices', str(devices),
           '--accelerator', accelerator,
           '--recycling_steps', str(recycling_steps),
           '--sampling_steps', str(sampling_steps),
           '--diffusion_samples', str(diffusion_samples),
           '--step_scale', str(step_scale),
           '--num_workers', str(num_workers),
           '--preprocessing-threads', str(preprocessing_threads),
           '--sampling_steps_affinity', str(sampling_steps_affinity),
           '--diffusion_samples_affinity', str(diffusion_samples_affinity),
           '--method', boltz_method]
    cmd += final_strs
    
    yield gr.update(value='Predicting...', interactive=False), ''
    full_output = ''
    curr_running_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8")
    for line in iter(curr_running_process.stdout.readline, ''):
        if 'The loaded checkpoint was produced with' in line or\
            'You are using a CUDA device' in line:  # Just skip these warnings
            continue
        if line.startswith('Predicting DataLoader'):
            full_output = full_output.rsplit('\n', 2)[0] + '\n' + line
        else:
            full_output += line
        yield gr.update(value='Predicting...', interactive=False), full_output
    curr_running_process.stdout.close()
    curr_running_process.wait()
    full_output += 'Prediction Done!\nWriting combined model...\n'
    out_struct_dir = Path(os.path.join(out_rng_dir, f'boltz_results_{file_name}', 'predictions', file_name))
    all_mdls = []
    if os.path.exists(out_struct_dir):
        for f in os.listdir(out_struct_dir):
            if f.endswith('.cif'):
                all_mdls.append(os.path.join(out_struct_dir, f))
        all_mdls = sorted(all_mdls, key=lambda x: int(x.rsplit('.', 1)[0].rsplit('_')[-1]))
        combined_cif_pth = os.path.join(out_struct_dir, f'{file_name}_model_combined.cif')
        combine_and_write_cif(all_mdls, combined_cif_pth)
        full_output += 'Combined model written!'
    else:
        full_output += 'Prediction failed!'
    
    yield gr.update(value='Run Boltz', interactive=True), full_output

def execute_multi_boltz(all_files: list[str],
                        devices: int, accelerator: str,
                        recycling_steps: int, sampling_steps: int,
                        diffusion_samples: float, step_scale: int,
                        num_workers: int, preprocessing_threads: int,
                        affinity_mw_correction: bool,
                        sampling_steps_affinity: int, diffusion_samples_affinity: int,
                        use_potentials: bool, boltz_method: str, no_kernels: bool, override: bool):
    # even though all the files are passed here, only their directory will be used 
    # since Boltz inherently allow batch processing
    dirname = os.path.dirname(all_files[0])
    rng_basename = os.path.basename(dirname)
    out_rng_dir = os.path.join(output_dir, rng_basename)
    check_dir_exist_and_rename(out_rng_dir)
    final_strs = ['--use_msa_server', '--write_full_pae', '--write_full_pde']
    if use_potentials:
        final_strs.append('--use_potentials')
    if affinity_mw_correction:
        final_strs.append('--affinity_mw_correction')
    if no_kernels:
        final_strs.append('--no_kernels')
    if override:
        final_strs.append('--override')
    cmd = ['boltz', 'predict', dirname,
           '--out_dir', out_rng_dir,
           '--devices', str(devices),
           '--accelerator', accelerator,
           '--recycling_steps', str(recycling_steps),
           '--sampling_steps', str(sampling_steps),
           '--diffusion_samples', str(diffusion_samples),
           '--step_scale', str(step_scale),
           '--num_workers', str(num_workers),
           '--preprocessing-threads', str(preprocessing_threads),
           '--sampling_steps_affinity', str(sampling_steps_affinity),
           '--diffusion_samples_affinity', str(diffusion_samples_affinity),
           '--method', boltz_method]
    cmd += final_strs
    
    yield gr.update(value='Predicting...', interactive=False), ''
    full_output = ''
    curr_running_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8")
    for line in iter(curr_running_process.stdout.readline, ''):
        if 'The loaded checkpoint was produced with' in line or\
            'You are using a CUDA device' in line:
            continue
        if line.startswith('Predicting DataLoader'):
            full_output = full_output.rsplit('\n', 2)[0] + '\n' + line
        else:
            full_output += line
        yield gr.update(value='Predicting...', interactive=False), full_output
    curr_running_process.stdout.close()
    curr_running_process.wait()
    full_output += 'Prediction Done!\n'
    
    out_pred_dir = Path(os.path.join(out_rng_dir, f'boltz_results_{rng_basename}', 'predictions'))
    dir_names_output_map = [{'out' : out_pred_dir/n/f'{n}_model_combined.cif',
                             'cifs': [str(out_pred_dir / n / _f) 
                                      for _f in sorted([f for f in os.listdir(out_pred_dir/n) if f.endswith('.cif')],
                                                       key=lambda x: int(x.rsplit('.', 1)[0].rsplit('_')[-1]))]} 
                            for n in os.listdir(out_pred_dir) if os.path.isdir(out_pred_dir / n)]
    progress_text = f'Writing combined model: 0/{len(dir_names_output_map)}'
    yield gr.update(), full_output + progress_text
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(combine_and_write_cif, d['cifs'], d['out']) for 
                   d in dir_names_output_map]
        total = len(futures)
        n = 0
        progress_text = f'Writing combined model: {n:>{len(str(total))}}/{total}'
        yield gr.update(), full_output + progress_text
        for f in as_completed(futures):
            n += 1
            progress_text = f'Writing combined model: {n:>{len(str(total))}}/{total}'
            yield gr.update(), full_output + progress_text
    
    full_output += f'{progress_text}\nCombined model written!'
    
    yield gr.update(value='Batch Predict', interactive=True), full_output

def execute_vhts_boltz(file_prefix: str, all_ligands: pd.DataFrame,
                       ligand_chain: str, yaml_str: str,
                       devices: int, accelerator: str,
                       recycling_steps: int, sampling_steps: int,
                       diffusion_samples: float, step_scale: int,
                       num_workers: int, preprocessing_threads: int,
                       affinity_mw_correction: bool,
                       sampling_steps_affinity: int, diffusion_samples_affinity: int,
                       use_potentials: bool, boltz_method: str, no_kernels: bool, override: bool):
    random_dir_name = f"{file_prefix}_vHTS_{uuid.uuid4().hex[:8]}"
    inp_rng_dir = os.path.join(input_dir, random_dir_name)
    out_rng_dir = os.path.join(output_dir, random_dir_name)
    check_dir_exist_and_rename(inp_rng_dir)
    check_dir_exist_and_rename(out_rng_dir)
    yaml_template_dict = safe_load(yaml_str)
    
    final_strs = ['--use_msa_server', '--write_full_pae', '--write_full_pde']
    if use_potentials:
        final_strs.append('--use_potentials')
    if affinity_mw_correction:
        final_strs.append('--affinity_mw_correction')
    if no_kernels:
        final_strs.append('--no_kernels')
    # Never override for vHTS
    # if override:
    #     final_strs.append('--override')
    cmd = ['boltz', 'predict', inp_rng_dir,
           '--out_dir', out_rng_dir,
           '--devices', '1',    # force use only single device
           '--accelerator', accelerator,
           '--recycling_steps', str(recycling_steps),
           '--sampling_steps', str(sampling_steps),
           '--diffusion_samples', str(diffusion_samples),
           '--step_scale', str(step_scale),
           '--num_workers', str(num_workers),
           '--preprocessing-threads', str(preprocessing_threads),
           '--sampling_steps_affinity', str(sampling_steps_affinity),
           '--diffusion_samples_affinity', str(diffusion_samples_affinity),
           '--method', boltz_method]
    cmd += final_strs
    
    for idx, row in all_ligands.iterrows():
        n, s = row['Name'], row['SMILES']
        for seq_info in yaml_template_dict['sequences']:
            if 'ligand' in seq_info and seq_info['ligand']['id'] == ligand_chain:
                seq_info['ligand']['smiles'] = s
                break
        inp_yaml_pth = os.path.join(inp_rng_dir, f'{n}.yaml')
        with open(inp_yaml_pth, 'w') as f:
            f.write(safe_dump(yaml_template_dict))
        
        # execute on only a single file to retrieve msa, prevent colabfold server overload
        if idx == 0:
            yield gr.update(value='Predicting...', interactive=False), ''
            full_output = ''
            curr_running_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                    text=True, encoding="utf-8")
            for line in iter(curr_running_process.stdout.readline, ''):
                if 'The loaded checkpoint was produced with' in line or\
                    'You are using a CUDA device' in line:  # Just skip these warnings
                    continue
                if line.startswith('Predicting DataLoader'):
                    full_output = full_output.rsplit('\n', 2)[0] + '\n' + line
                else:
                    full_output += line
                yield gr.update(value='Predicting...', interactive=False), full_output
            curr_running_process.stdout.close()
            curr_running_process.wait()
            num_msa_pth_map = {}
            msa_dir = os.path.join(out_rng_dir, f'boltz_results_{random_dir_name}', 'msa')
            for msa_f in os.listdir(msa_dir):
                if msa_f.endswith('.csv'):
                    num = msa_f.rsplit('.', 1)[0].rsplit('_', 1)[-1]
                    num_msa_pth_map[int(num)] = os.path.join(msa_dir, msa_f)
            # Just add the csv path containing the MSA to the "msa" key of template. 
            # Number by the index of list within the "sequences" key!
            for seq_num, seq_info in enumerate(yaml_template_dict['sequences']):
                if seq_num in num_msa_pth_map:
                    seq_info['protein']['msa'] = num_msa_pth_map[seq_num]
    
    cmd.remove('--use_msa_server')
    cmd[6] = str(devices)   # replace the "devices" param back to user-defined value
    
    curr_running_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8")
    for line in iter(curr_running_process.stdout.readline, ''):
        if 'The loaded checkpoint was produced with' in line or\
            'You are using a CUDA device' in line:
            continue
        if line.startswith('Predicting DataLoader'):
            full_output = full_output.rsplit('\n', 2)[0] + '\n' + line
        else:
            full_output += line
        yield gr.update(), full_output
    curr_running_process.stdout.close()
    curr_running_process.wait()
    full_output += 'Prediction Done. Post-Processing files...\n'
    yield gr.update(), full_output
    
    out_pred_dir = Path(os.path.join(out_rng_dir, f'boltz_results_{random_dir_name}', 'predictions'))
    dir_smiles_dict = {}
    for _, row in all_ligands.iterrows():
        name, smiles = row['Name'], row['SMILES']
        dir_smiles_dict[out_pred_dir / f'{name}'] = smiles
    dir_names_output_map = [{'out' : out_pred_dir/n/f'{n}_model_combined.cif',
                             'cifs': [str(out_pred_dir / n / _f) 
                                      for _f in sorted([f for f in os.listdir(out_pred_dir/n) if f.endswith('.cif')],
                                                       key=lambda x: int(x.rsplit('.', 1)[0].rsplit('_')[-1]))],
                             'smiles': dir_smiles_dict[out_pred_dir/n]} 
                            for n in os.listdir(out_pred_dir) if os.path.isdir(out_pred_dir / n)]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(recover_and_combine_cif, d['cifs'], d['smiles'], ligand_chain, d['out']) for 
                   d in dir_names_output_map]
        total = len(futures)
        n = 0
        progress_text = f'Post-Processing Progress: {n:>{len(str(total))}} / {total}'
        yield gr.update(), full_output + progress_text
        for f in as_completed(futures):
            err = f.result()
            n += 1
            progress_text = f'Post-Processing Progress: {n:>{len(str(total))}} / {total}'
            yield gr.update(), full_output + progress_text
            
    progress_text += '\nvHTS done!'
    yield gr.update(value='Run vHTS', interactive=True), full_output + progress_text

### vHTS ###
def update_chem_file_format(chem_type: str):
    if chem_type == 'Chemical files':
        file_types=['.sdf', '.mol', '.smi', '.zip']
        label = 'Upload chemical file(s)'
        tabular_visibility = gr.update(visible=False)
    elif chem_type == 'Tabular files':
        file_types = ['.csv', '.tsv', '.txt']
        label = 'Upload tabular file(s)'
        tabular_visibility = gr.update(visible=True)
    return (gr.update(file_types=file_types, label=label), tabular_visibility,
            tabular_visibility, tabular_visibility)

def __check_smi_title_line(smi_file: str):
    with open(smi_file) as f:
        for r, l in enumerate(f):
            possible_smiles = l.split(' ')[0]
            if Chem.MolFromSmiles(possible_smiles) is not None:
                return r
        return 0

def _process_single_chem_file(chem_f: str):
    if chem_f.endswith('.sdf'):
        mols = Chem.MultithreadedSDMolSupplier(chem_f)
    elif chem_f.endswith('.mol'):
        mols = [Chem.MolFromMolFile(chem_f)]
    elif chem_f.endswith('.smi'):
        n = __check_smi_title_line(chem_f)
        mols = Chem.MultithreadedSmilesMolSupplier(chem_f, titleLine=n)
    names, smiles = [], []
    for mol in mols:
        if mol is None:
            continue
        if mol.HasProp('_Name'):
            name = mol.GetProp('_Name')
        else:
            name = os.path.basename(chem_f).rsplit('.', 1)[0]
        smi = Chem.MolToSmiles(mol)
        names.append(name)
        smiles.append(smi)
    return names, smiles

def _process_uploaded_chem_file(f: str):
    if f.endswith(('.sdf', '.mol', '.smi')):
        final_names, final_smiles = _process_single_chem_file(f)
    elif f.endswith('.zip'):
        with zipfile.ZipFile(f, 'r') as zip_ref:
            final_names, final_smiles = [], []
            for filename in zip_ref.namelist():
                if filename.endswith(('.sdf', '.mol', '.smi', '.zip')):
                    with zip_ref.open(filename) as file_in_zip:
                        file_content = file_in_zip.read().decode()
                        with tempfile.NamedTemporaryFile(suffix='.'+filename.rsplit('.', 1)[-1], delete=False) as temp_file:
                            temp_file.write(file_content.encode('utf-8'))
                            temp_file.flush()
                            temp_file_path = temp_file.name
                        extracted_n, extracted_s = _process_uploaded_chem_file(temp_file_path)
                        os.remove(temp_file_path)
                        final_names.extend(extracted_n)
                        final_smiles.extend(extracted_s)
    return [final_names, final_smiles]

def _process_tabular_files(chem_f: list[str], name_col: str, chem_col: str, delimiter: str):
    try:
        df = pd.read_csv(chem_f, delimiter=delimiter)
        if name_col in df and chem_col in df:
            df = df[[name_col, chem_col]].dropna()
        else:
            return [], []
    except:
        return [], []
    final_names, final_smiles = [], []
    for _, row in df.iterrows():
        name = row[name_col]
        chem_str = row[chem_col]
        if chem_str.startswith('InChI='):
            mol = Chem.MolFromInchi(chem_str)
        else:
            mol = Chem.MolFromSmiles(chem_str)
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            final_names.append(name)
            final_smiles.append(smi)
    return final_names, final_smiles

def process_uploaded_ligand(chem_files: list[str], name_col: str,
                            chem_col: str, delimiter: str, prev_df: pd.DataFrame):
    total_names, final_smiles = prev_df['Name'].to_list(), prev_df['SMILES'].to_list()
    for chem_f in chem_files:
        if chem_f.endswith(('.sdf', '.mol', '.smi', '.zip')):
            a, b = _process_uploaded_chem_file(chem_f)
        elif chem_f.endswith(('.csv', '.tsv', '.txt')):
            a, b = _process_tabular_files(chem_f, name_col, chem_col, delimiter)
        total_names.extend(a)
        final_smiles.extend(b)
    final_names = []
    for name in total_names:
        if name in final_names:
            i = 2
            new_name = f'{name}_{i}'
            while new_name in final_names:
                i += 1
                new_name = f'{name}_{i}'
            name = new_name
        final_names.append(name)
    return pd.DataFrame({'Name': final_names, 'SMILES': final_smiles})

def __extract_ligand_coord(cif_pth: str, lig_chain: str):
    p_map = {'Chain': 0, 'Atom': 0, 'X': 0, 'Y': 0, 'Z': 0}
    atom_coord_info = []
    
    n = -1
    with open(cif_pth) as f:
        for l in f:
            if l.startswith('_atom_site.'):
                n += 1
            if l.startswith('_atom_site.auth_asym_id'):
                p_map['Chain'] = n
            elif l.startswith('_atom_site.type_symbol'):
                p_map['Atom'] = n
            elif l.startswith('_atom_site.Cartn_x'):
                p_map['X'] = n
            elif l.startswith('_atom_site.Cartn_y'):
                p_map['Y'] = n
            elif l.startswith('_atom_site.Cartn_z'):
                p_map['Z'] = n
            
            if l.startswith('HETATM'):
                line_splitted = l.split()
                if line_splitted[p_map['Chain']] == lig_chain:
                    a, x, y, z = line_splitted[p_map['Atom']], line_splitted[p_map['X']], \
                        line_splitted[p_map['Y']], line_splitted[p_map['Z']]
                    a = Chem.Atom(periodic_table.GetAtomicNumber(a.lower().capitalize()))
                    atom_coord_info.append((a, Point3D(float(x), float(y), float(z))))
            if atom_coord_info and l.startswith('#'):
                break
    return atom_coord_info

def __reconstruct_mol_from_data(mol_data: list[tuple]):
    mol = Chem.EditableMol(Chem.Mol())
    conf = Chem.Conformer(len(mol_data))
    fc = 0
    for i, data in enumerate(mol_data):
        atom, coord = data
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, coord)
        fc += atom.GetFormalCharge()
    mol = mol.GetMol()
    mol.AddConformer(conf)
    DetermineConnectivity(mol)
    return mol

def recover_and_combine_cif(cif_files: list, smiles: str, ligand_chain: str, out_cif: str):
    ref_mol = Chem.MolFromSmiles(smiles)
    errors = ''
    final_mols = []
    dir_name = os.path.basename(os.path.dirname(cif_files[0]))
    for f in cif_files:
        try:
            data = __extract_ligand_coord(f, ligand_chain)
            coord_mol = __reconstruct_mol_from_data(data)
            final_mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, coord_mol)
            AllChem.AssignStereochemistryFrom3D(final_mol)
            for atom in final_mol.GetAtoms():
                atom.SetNoImplicit(False)
                atom.SetNumRadicalElectrons(0)
            final_mol.UpdatePropertyCache()
            Chem.SanitizeMol(final_mol)
            name = os.path.basename(f).rsplit('.', 1)[0]
            out_sdf_f = os.path.join(os.path.dirname(f), name + '.sdf')
            final_mol.SetProp('_Name', name)
            final_mol.SetProp('SMILES', Chem.MolToSmiles(final_mol))
            with Chem.SDWriter(out_sdf_f) as w:
                w.write(final_mol)
            final_mols.append(final_mol)
        except Exception as e:
            print(e)
            errors += f'{e}\n'
    csv_name = os.path.join(os.path.dirname(f), dir_name + '_bust.csv')
    target_col = ['mol_pred_loaded', 'sanitization', 'all_atoms_connected', 'bond_lengths', 'bond_angles',
                  'internal_steric_clash', 'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness',
                  'double_bond_flatness', 'internal_energy', 'passes_valence_checks', 'passes_kekulization']
    total_files = len(cif_files)
    if final_mols:
        buster = PoseBusters('mol')
        df = buster.bust(final_mols, full_report=True)
        df = df.reset_index()
        full_cols = df.columns
        df['All Passes'] = df[target_col].sum(1) == len(target_col)
        df['Rank'] = df['molecule'].apply(lambda x: int(x.rsplit('_', 1)[-1])+1)
        all_ranks = df['Rank'].to_list()
        
        for i in range(1, total_files+1):
            if i not in all_ranks:
                new_row = {c: np.nan for c in full_cols}
                new_row.update({'Rank': i, 'All Passes': False})
                df.loc[len(df)] = new_row
        df = df.sort_values('Rank')
        
        final_col = ['Rank'] + target_col + ['All Passes']
        df = df[final_col].astype({c: 'boolean' for c in final_col if c != 'Rank'})
    else:
        rows = {c: [np.nan]*total_files for c in target_col}
        rows['Rank'] = [i for i in range(1, total_files+1)]
        rows['All Passes'] = [False] * total_files
        df = pd.DataFrame(rows)
    df.to_csv(csv_name, index=False)
    combine_and_write_cif(cif_files, out_cif)
    return errors

### Result visulization ###
def get_general_molstar_html(mmcif_base64, mdl_idx, color='chain-id'):
    return f"""
    <iframe
        id="molstar_frame_general"
        style="width: 100%; height: 600px; border: none;"
        srcdoc='
            <!DOCTYPE html>
            <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
                </head>
                <body>
                    <div id="Viewer" style="width: 1200px; height: 400px; position: center"></div>
                    <script>
                        (async function() {{
                            const viewer = new rcsbMolstar.Viewer("Viewer",
                            {{layoutIsExpanded: true,
                              layoutShowControls: false,
                              viewportShowExpand: true,
                              showWelcomeToast: false}});
                            
                            const mmcifBase64 = "{mmcif_base64}";
                            const rawString = atob(mmcifBase64);
                            
                            const data = await viewer.plugin.builders.data.rawData({{
                                data: rawString,
                                label: "mmcif"
                                }});
                            
                            const trajectory = await viewer.plugin.builders.structure.parseTrajectory(data, "mmcif");
                            const repr = await viewer.plugin.builders.structure.hierarchy.applyPreset(
                                trajectory,
                                "default",
                                {{ model: {{ modelIndex: {mdl_idx} }}, 
                                   representationPresetParams: {{ theme: {{ globalName: "{color}" }} }} 
                                }})
                            
                            window.trajectory = trajectory;
                            window.viewer     = viewer;
                            window.modelIndex = {mdl_idx};
                            window.repr = repr;
                            window.addEventListener("message", async (e) => {{
                                if (e.data?.type === "change-theme") {{
                                    window.parent.console.log(e.data.themeName);
                                    window.parent.console.log( repr );
                                }}
                            }});
                      }})();
                    </script>
                </body>
            </html>
        '>
    </iframe>"""

def read_output_files(read_vhts: bool):
    name_rank_map = {}
    for out_f_or_d in os.listdir(output_dir):
        rng_dir = os.path.join(output_dir, out_f_or_d)
        is_vhts = out_f_or_d.rsplit('_')[-2] == 'vHTS'
        if not read_vhts and is_vhts:
            continue
        if os.path.isdir(rng_dir):
            for target_pth in os.listdir(rng_dir):
                if not target_pth.startswith('boltz_results_'):
                    continue
                target_dir = os.path.join(rng_dir, target_pth)
                pred_parent_dir = os.path.join(target_dir, 'predictions')
                for name in os.listdir(pred_parent_dir):
                    pred_dir = os.path.join(pred_parent_dir, name)
                    if not os.path.isdir(pred_dir):
                        continue
                    if name in name_rank_map:
                        i = 2
                        new_name = f'{name}_{i}'
                        while new_name in name_rank_map:
                            i += 1
                            new_name = f'{name}_{i}'
                    else:
                        new_name = name
                    name_rank_map[new_name] = []
                    all_files = os.listdir(pred_dir)
                    total_models = len(all_files) // 5
                    aff_pth = os.path.join(pred_dir, f'affinity_{name}.json')
                    combined_cif_pth = os.path.join(pred_dir, f'{name}_model_combined.cif')
                    if not os.path.exists(aff_pth):
                        aff_pth = None
                    if not os.path.exists(combined_cif_pth):
                        # Default back to using single model.
                        # This shouldn't happen unless user decide to load the structure 
                        # before the entire prediction is done.
                        combined_cif_pth = None
                    for i in range(total_models):
                        cnf_pth  = os.path.join(pred_dir, f'confidence_{name}_model_{i}.json')
                        mdl_pth  = os.path.join(pred_dir, f'{name}_model_{i}.cif') if combined_cif_pth is None else combined_cif_pth
                        pae_pth  = os.path.join(pred_dir, f'pae_{name}_model_{i}.npz')
                        pde_pth  = os.path.join(pred_dir, f'pde_{name}_model_{i}.npz')
                        plddt_pth  = os.path.join(pred_dir, f'plddt_{name}_model_{i}.npz')
                        name_rank_map[new_name].append({'confidence': cnf_pth,
                                                        'affinity'  : aff_pth,
                                                        'cif_model' : mdl_pth,
                                                        'pae_pth'   : pae_pth,
                                                        'pde_pth'   : pde_pth,
                                                        'plddt_pth' : plddt_pth})
    return name_rank_map

def update_output_name_dropdown(read_vhts: bool):
    name_rank_f_map = read_output_files(read_vhts)
    return (gr.update(choices=list(name_rank_f_map)),
            gr.update(choices=['Rank 1'], value='Rank 1'),
            name_rank_f_map)

def update_name_rank_dropdown(name: str, name_rank_f_map: dict):
    total_rank = len(name_rank_f_map[name])
    return gr.update(choices=[f'Rank {i}' for i in range(1, total_rank + 1)])

def update_result_visualization(name: str, rank_name: str, name_rank_f_map: dict, color: str):
    if not rank_name.strip():
        return [gr.update()] * 8
    rank = int(rank_name.split(' ')[-1]) - 1
    conf_metrics = name_rank_f_map[name][rank]
    if rank+1 > len(conf_metrics):
        return [gr.update()] * 8
    with open(conf_metrics['cif_model']) as f:
        mdl_strs = f.read()
    cif_base64 = base64.b64encode(mdl_strs.encode()).decode('utf-8')
    
    yield (get_general_molstar_html(cif_base64, rank, color), gr.update(''), gr.update(''),
           gr.update(''), gr.update(''), gr.update(''), gr.update(''), gr.update(''))
    
    with open(conf_metrics['confidence']) as f:
        conf_dict = json.load(f)
    overall_conf, chain_conf, pair_chain_conf = [], [], []
    for conf_id, value in conf_dict.items():
        if isinstance(value, float):
            overall_conf.append([conf_id, f'{value:.4f}'])
        elif conf_id == 'chains_ptm':
            for chain_n, ptm_value in value.items():
                chain_conf.append([f'{int(chain_n)+1}', f'{ptm_value:.4f}'])
        elif conf_id == 'pair_chains_iptm':
            for i, all_ptm_value in enumerate(value.values()):
                pair_chain_conf.append([])
                for single_ptm_value in all_ptm_value.values():
                    pair_chain_conf[i].append(f'{single_ptm_value:.4f}')
    aff_f = conf_metrics['affinity']
    if aff_f is not None:
        aff_update = []
        with open(aff_f) as f:
            aff_data = json.load(f)
        for aff_metric, aff_value in aff_data.items():
            aff_update.append([aff_metric, f'{aff_value:.4f}'])
        # combined_score = max((-aff_data['affinity_pred_value']-2)/4, 0) * aff_data['affinity_probability_binary']
        # aff_update.append(['Overall Score', f'{combined_score:.4f}'])
        aff_update = gr.update(value=aff_update, visible=True)
    else:
        aff_update = gr.update(visible=False)
    
    length_split = [0]
    chain_entity_map = {}
    res_id_idx, entity_id_idx, chain_id_idx, mdl_num_idx = 0, 0, 0, 0
    idx = -1
    if mdl_strs is not None:
        last_res, last_c, i, last_mdl_num = None, None, 0, None
        for line in mdl_strs.split('\n'):
            if line.startswith(('_atom_site.')):
                idx += 1
                if   line.endswith('.label_seq_id'):
                    res_id_idx = idx
                elif line.endswith('.label_entity_id'):
                    entity_id_idx = idx
                elif line.endswith('.label_asym_id'):
                    chain_id_idx = idx
                elif line.endswith('.pdbx_PDB_model_num'):
                    mdl_num_idx = idx
            elif line.startswith(('ATOM', 'HETATM')):
                if line.strip() == '#':
                    break
                all_splitted = line.strip().split()
                res_id, entity_id, c, mdl_num = all_splitted[res_id_idx], all_splitted[entity_id_idx], \
                    all_splitted[chain_id_idx], all_splitted[mdl_num_idx]
                chain_entity_map[c] = entity_id
                if last_c is not None and last_c != c:
                    length_split.append(int(last_res) if last_res != '.' else i)
                    i = 0
                if last_mdl_num is not None and mdl_num != last_mdl_num:
                    break
                last_c = c
                last_res = res_id
                last_mdl_num = mdl_num
                if res_id == '.':
                    i += 1
            elif line == '_atom_type.symbol':
                if last_c is not None:
                    length_split.append(int(last_res) if last_res != '.' else i)
                break
    
    length_split = np.cumsum(length_split)
    pae_mat = np.load(conf_metrics['pae_pth'])['pae']
    pde_mat = np.load(conf_metrics['pde_pth'])['pde']
    plddt_array = np.load(conf_metrics['plddt_pth'])['plddt']
    total_length = pae_mat.shape[0]
    pae_fig = px.imshow(pae_mat, color_continuous_scale='Greens_r',
                        range_color=[0.25, 31.75], labels={'color': 'PAE (Å)'})
    for i in range(len(length_split)-2):
        end = length_split[i+1]
        pae_fig.add_shape(type='line', x0=0, y0=end-0.5,
                          x1=total_length-0.5, y1=end-0.5,
                          line=dict(color="Cornflowerblue"))
        pae_fig.add_shape(type='line', x0=end-0.5, y0=0,
                          x1=end-0.5, y1=total_length-0.5,
                          line=dict(color="Cornflowerblue"))
    pde_fig = px.imshow(pde_mat, color_continuous_scale='Greens_r',
                        range_color=[0.25, 31.75], labels={'color': 'PDE (Å)'})
    for i in range(len(length_split)-2):
        end = length_split[i+1]
        pde_fig.add_shape(type='line', x0=0, y0=end-0.5,
                          x1=total_length-0.5, y1=end-0.5,
                          line=dict(color="Cornflowerblue"))
        pde_fig.add_shape(type='line', x0=end-0.5, y0=0,
                          x1=end-0.5, y1=total_length-0.5,
                          line=dict(color="Cornflowerblue"))
    plddt_fig = go.Figure()
    all_chains = list(chain_entity_map)
    for i in range(len(length_split)-1):
        curr_c = all_chains[i]
        splitted_plddt = plddt_array[length_split[i]:length_split[i+1]]
        x_vals = np.arange(length_split[i]+1, length_split[i+1]+1)
        mode = 'lines' if splitted_plddt.shape[0] > 1 else 'markers'
        plddt_fig.add_trace(go.Scatter(x=x_vals,
                                       y=splitted_plddt,
                                       mode=mode,
                                       name=f'Chain {curr_c} (Entity {chain_entity_map[curr_c]})'))
    pae_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    pde_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    plddt_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(title=dict(text='Residue')),
                            yaxis=dict(title=dict(text='pLDDT'), range=[0, 1]),
                            template='simple_white')
    yield [gr.update(), overall_conf, chain_conf,
            gr.DataFrame(value=pair_chain_conf,
                         headers=[f'{i+1}' for i in range(len(chain_conf))],
                         show_row_numbers=True, column_widths=['30px'] * len(chain_conf)),
            aff_update, pae_fig, pde_fig, plddt_fig]

def update_general_molstar_only(name: str, rank_name: str, name_rank_f_map: dict, color: str):
    if not rank_name.strip():
        return [gr.update()] * 8
    rank = int(rank_name.split(' ')[-1]) - 1
    conf_metrics = name_rank_f_map[name][rank]
    with open(conf_metrics['cif_model']) as f:
        mdl_strs = f.read()
    cif_base64 = base64.b64encode(mdl_strs.encode()).decode('utf-8')
    
    yield get_general_molstar_html(cif_base64, rank, color)

### vHTS Processing ###
def get_vhts_molstar_html(mmcif_base64, mdl_idx, color='plddt-confidence'):
    return f"""
    <iframe
        id="molstar_frame_general"
        style="width: 100%; height: 600px; border: none;"
        srcdoc='
            <!DOCTYPE html>
            <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
                </head>
                <body>
                    <div id="Viewer" style="width: 1200px; height: 400px; position: center"></div>
                    <script>
                        (async function() {{
                            const viewer = new rcsbMolstar.Viewer("Viewer",
                            {{layoutIsExpanded: true,
                              layoutShowControls: false,
                              viewportShowExpand: true,
                              showWelcomeToast: false}});
                            
                            const mmcifBase64 = "{mmcif_base64}";
                            const rawString = atob(mmcifBase64);
                            
                            const data = await viewer.plugin.builders.data.rawData({{
                                data: rawString,
                                label: "mmcif"
                                }});
                            
                            const trajectory = await viewer.plugin.builders.structure.parseTrajectory(data, "mmcif");
                            await viewer.plugin.builders.structure.hierarchy.applyPreset(
                                trajectory,
                                "default",
                                {{ model: {{ modelIndex: {mdl_idx} }}, 
                                   representationPresetParams: {{ theme: {{ globalName: "{color}" }} }} 
                                }})
                            window.trajectory = trajectory;
                            window.viewer     = viewer;
                            window.modelIndex = {mdl_idx};
                            window.addEventListener("message", async (e) => {{
                                if (e.data?.type === "change-theme") {{
                                    await viewer.plugin.builders.structure.hierarchy.applyPreset(
                                        trajectory, "default",
                                        {{
                                            model: {{ modelIndex: window.modelIndex }}, 
                                            representationPresetParams: {{
                                                theme: {{ globalName: e.data.themeName }}
                                            }}
                                        }}
                                    );
                                }}
                            }});
                      }})();
                    </script>
                </body>
            </html>
        '>
    </iframe>"""

def read_vhts_directory():
    vhts_name_df = {}
    vhts_name_pth_map = {}
    for out_f_or_d in os.listdir(output_dir):
        rng_dir = os.path.join(output_dir, out_f_or_d)
        # If 'vHTS' is at the end of user-defined complex name then this will match too.
        # Make a new dir specifically for vHTS in the future?
        is_vhts = out_f_or_d.rsplit('_')[-2] == 'vHTS'
        if os.path.isdir(rng_dir) and is_vhts:
            for target_pth in os.listdir(rng_dir):
                if 'boltz_results_' not in target_pth:
                    continue
                name = target_pth.split('boltz_results_', 1)[-1].rsplit('_', 2)[0]
                if name in vhts_name_df:
                    i = 2
                    new_name = f'{name}_{i}'
                    while new_name in vhts_name_df:
                        i += 1
                        new_name = f'{name}_{i}'
                    name = new_name
                pred_dir = os.path.join(rng_dir, target_pth, 'predictions')
                df_data = {'Name': [], 'ligand ipTM': [], 'binding prob.': [],
                           'binding aff.': [], 'Pass Posebusters?': []}
                vhts_name_pth_map[name] = {'Name': [], 'conf': [], 'aff': [],
                                           'struct': [], 'pae': [], 'pde': [], 'plddt': []}
                for n in os.listdir(pred_dir):
                    if n.startswith('.'):
                        continue
                    docked_dir = Path(os.path.join(pred_dir, n))
                    combined_cif_pth = os.path.join(docked_dir, f'{n}_model_combined.cif')
                    if not os.path.exists(combined_cif_pth):    # Same logic as the general result visualization
                        combined_cif_pth = os.path.join(docked_dir, f'{n}_model_0.cif')
                    if os.path.isdir(docked_dir):
                        conf_pth   = docked_dir / f'confidence_{n}_model_0.json'
                        aff_pth    = docked_dir / f'affinity_{n}.json'
                        struct_pth = docked_dir / combined_cif_pth
                        pae_pth    = docked_dir / f'pae_{n}_model_0.npz'
                        pde_pth    = docked_dir / f'pde_{n}_model_0.npz'
                        plddt_pth  = docked_dir / f'plddt_{n}_model_0.npz'
                        pose_bust  = docked_dir / f'{n}_bust.csv'
                        with open(conf_pth) as f:
                            ligand_iptm = json.load(f)['ligand_iptm']
                        with open(aff_pth) as f:
                            aff_data = json.load(f)
                            binding_aff = aff_data['affinity_pred_value']
                            binding_prob = aff_data['affinity_probability_binary']
                        if os.path.isfile(pose_bust):
                            df = pd.read_csv(pose_bust)
                            rank_1_pass = df[df['Rank'] == 1]['All Passes'].to_list()[0]
                        else:
                            rank_1_pass = float('nan')
                            pose_bust = None
                        for k, v in zip(df_data, [n, ligand_iptm, binding_prob, binding_aff, rank_1_pass]):
                            df_data[k].append(v)
                        vhts_name_pth_map[name][n] = {'conf'  : conf_pth,
                                                      'aff'   : aff_pth,
                                                      'struct': struct_pth,
                                                      'pae'   : pae_pth,
                                                      'pde'   : pde_pth,
                                                      'plddt' : plddt_pth,
                                                      'pose'  : pose_bust,}
                df_data['Parent'] = [name] * len(df_data['Name'])
                vhts_name_df[name] = pd.DataFrame(df_data)
    return vhts_name_df, vhts_name_pth_map, gr.update(choices=list(vhts_name_df), value=None)

def update_vhts_df_with_selects(names: list[str], name_df_map: dict):
    if not names:
        return pd.DataFrame()
    return pd.concat([name_df_map[n] for n in names]).reset_index(drop=True)

def update_vhts_result_visualization(name_fpth_map: dict, evt: gr.SelectData):
    row_value = evt.row_value
    if not row_value[0]:
        yield [gr.update()] * 9 + [f'<span style="font-size:15px; font-weight:bold;">Failed to load visualization</span>']
    
    parent, name = row_value[-1], row_value[0]
    conf_metrics = name_fpth_map[parent][name]
    with open(conf_metrics['struct']) as f:
        mdl_strs = f.read()
    cif_base64 = base64.b64encode(mdl_strs.encode()).decode('utf-8')
    yield [get_vhts_molstar_html(cif_base64, 0, 'plddt-confidence')] + [gr.update()] * 8 + [f'<span style="font-size:15px; font-weight:bold;">Visualization of {name}</span>']
    
    with open(conf_metrics['conf']) as f:
        conf_dict = json.load(f)
    overall_conf, chain_conf, pair_chain_conf = [], [], []
    for conf_id, value in conf_dict.items():
        if isinstance(value, float):
            overall_conf.append([conf_id, f'{value:.4f}'])
        elif conf_id == 'chains_ptm':
            for chain_n, ptm_value in value.items():
                chain_conf.append([f'{int(chain_n)+1}', f'{ptm_value:.4f}'])
        elif conf_id == 'pair_chains_iptm':
            for i, all_ptm_value in enumerate(value.values()):
                pair_chain_conf.append([])
                for single_ptm_value in all_ptm_value.values():
                    pair_chain_conf[i].append(f'{single_ptm_value:.4f}')
    aff_f = conf_metrics['aff']
    aff_update = []
    with open(aff_f) as f:
        aff_data = json.load(f)
    for aff_metric, aff_value in aff_data.items():
        aff_update.append([aff_metric, f'{aff_value:.4f}'])
    aff_update = gr.update(value=aff_update, visible=True)
    
    poseb_f = conf_metrics['pose']
    if poseb_f is not None:
        pose_df = pd.read_csv(poseb_f)
    else:
        pose_df = None
    
    yield [gr.update(), overall_conf, chain_conf,
           gr.DataFrame(value=pair_chain_conf,
                        headers=[f'{i+1}' for i in range(len(chain_conf))],
                        show_row_numbers=True, column_widths=['30px'] * len(chain_conf)),
           aff_update, gr.DataFrame(value=pose_df)] + [gr.update()] * 4
    
    length_split = [0]
    chain_entity_map = {}
    res_id_idx, entity_id_idx, chain_id_idx, mdl_num_idx = 0, 0, 0, 0
    idx = -1
    if mdl_strs is not None:
        last_res, last_c, i, last_mdl_num = None, None, 0, None
        for line in mdl_strs.split('\n'):
            if line.startswith(('_atom_site.')):
                idx += 1
                if   line.endswith('.label_seq_id'):
                    res_id_idx = idx
                elif line.endswith('.label_entity_id'):
                    entity_id_idx = idx
                elif line.endswith('.label_asym_id'):
                    chain_id_idx = idx
                elif line.endswith('.pdbx_PDB_model_num'):
                    mdl_num_idx = idx
            elif line.startswith(('ATOM', 'HETATM')):
                if line.strip() == '#':
                    break
                all_splitted = line.strip().split()
                res_id, entity_id, c, mdl_num = all_splitted[res_id_idx], all_splitted[entity_id_idx], \
                    all_splitted[chain_id_idx], all_splitted[mdl_num_idx]
                chain_entity_map[c] = entity_id
                if last_c is not None and last_c != c:
                    length_split.append(int(last_res) if last_res != '.' else i)
                    i = 0
                if last_mdl_num is not None and mdl_num != last_mdl_num:
                    break
                last_c = c
                last_res = res_id
                last_mdl_num = mdl_num
                if res_id == '.':
                    i += 1
            elif line == '_atom_type.symbol':
                if last_c is not None:
                    length_split.append(int(last_res) if last_res != '.' else i)
                break
    
    length_split = np.cumsum(length_split)
    pae_mat = np.load(conf_metrics['pae'])['pae']
    pde_mat = np.load(conf_metrics['pde'])['pde']
    total_length = pae_mat.shape[0]
    plddt_array = np.load(conf_metrics['plddt'])['plddt']
    pae_fig = px.imshow(pae_mat, color_continuous_scale='Greens_r',
                        range_color=[0.25, 31.75], labels={'color': 'PAE (Å)'})
    for i in range(len(length_split)-2):
        end = length_split[i+1]
        pae_fig.add_shape(type='line', x0=0, y0=end-0.5,
                          x1=total_length-0.5, y1=end-0.5,
                          line=dict(color="Cornflowerblue"))
        pae_fig.add_shape(type='line', x0=end-0.5, y0=0,
                          x1=end-0.5, y1=total_length-0.5,
                          line=dict(color="Cornflowerblue"))
    pde_fig = px.imshow(pde_mat, color_continuous_scale='Greens_r',
                        range_color=[0.25, 31.75], labels={'color': 'PDE (Å)'})
    for i in range(len(length_split)-2):
        end = length_split[i+1]
        pde_fig.add_shape(type='line', x0=0, y0=end-0.5,
                          x1=total_length-0.5, y1=end-0.5,
                          line=dict(color="Cornflowerblue"))
        pde_fig.add_shape(type='line', x0=end-0.5, y0=0,
                          x1=end-0.5, y1=total_length-0.5,
                          line=dict(color="Cornflowerblue"))
    plddt_fig = go.Figure()
    all_chains = list(chain_entity_map)
    for i in range(len(length_split)-1):
        curr_c = all_chains[i]
        splitted_plddt = plddt_array[length_split[i]:length_split[i+1]]
        x_vals = np.arange(length_split[i]+1, length_split[i+1]+1)
        mode = 'lines' if splitted_plddt.shape[0] > 1 else 'markers'
        plddt_fig.add_trace(go.Scatter(x=x_vals,
                                       y=splitted_plddt,
                                       mode=mode,
                                       name=f'Chain {curr_c} (Entity {chain_entity_map[curr_c]})'))
    pae_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    pde_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    plddt_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(title=dict(text='Residue')),
                            yaxis=dict(title=dict(text='pLDDT'), range=[0, 1]),
                            template='simple_white')
    yield [gr.update()] * 6 + [pae_fig, pde_fig, plddt_fig, 
                               f'<span style="font-size:15px; font-weight:bold;">Visualization of {name}</span>']

def download_vhts_dataframe(inp_df: pd.DataFrame, format: str):
    inp_df = pd.DataFrame(inp_df)
    if format == 'CSV':
        temp_dir = tempfile.gettempdir()
        saved_pth = os.path.join(temp_dir, 'vHTS_result.csv')
        inp_df.to_csv(saved_pth, index=False)
    elif format == 'TSV':
        temp_dir = tempfile.gettempdir()
        saved_pth = os.path.join(temp_dir, 'vHTS_result.tsv')
        inp_df.to_csv(saved_pth, index=False, sep='\t')
    elif format == 'XLSX':
        temp_dir = tempfile.gettempdir()
        saved_pth = os.path.join(temp_dir, 'vHTS_result.xlsx')
        inp_df.to_excel(saved_pth, index=False)
    return gr.update(saved_pth, interactive=True)

### Download Output ###
def zip_selected_files(all_files_and_dirs: list, zipname_pth_map: dict):
    rng_name = uuid.uuid4().hex[:8]
    zipped_file = os.path.join(curr_dir, f'output_{rng_name}.zip')
    final_files = []
    for f_or_d in all_files_and_dirs:
        if os.path.isfile(f_or_d):
            final_files.append(f_or_d)
    max_f_cnt_len = len(str(len(final_files)))
    yield f'{0:{max_f_cnt_len}}/{len(final_files)}', gr.update(), gr.update()
    c = 0
    with zipfile.ZipFile(zipped_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_f:
        for file in final_files:
            zip_f.write(file, os.path.relpath(file, output_dir))
            c += 1
            yield f'{c:{max_f_cnt_len}}/{len(final_files)}', gr.update(), gr.update()
    zipname_pth_map[os.path.basename(zipped_file)] = zipped_file
    yield 'Zipping done', zipped_file, zipname_pth_map

def zip_selected_option_files(names: list, name_pth_map: dict, zipname_pth_map: dict, options: list[str]):
    if not options:
        return f'No options selected!', None, gr.update()
    rng_name = uuid.uuid4().hex[:8]
    zipped_file = os.path.join(curr_dir, f'output_{rng_name}.zip')
    final_files = []
    prefixes, suffixes = [], []
    for option in options:
        if option == 'MSA':
            continue
        m = file_extract_matching_map[option]
        if m[0].startswith('.'):
            suffixes.extend(m)
        elif m[0].endswith('_'):
            prefixes.extend(m)
    regex_patterns = []
    if prefixes:
        regex_patterns.append(rf"^({'|'.join(prefixes)})")
    if suffixes:
        escaped_suffixes = [s.replace('.', r'\.') for s in suffixes]
        regex_patterns.append(rf"({'|'.join(escaped_suffixes)})$")
    if regex_patterns:
        file_pattern = re.compile('|'.join(regex_patterns))
    else:
        file_pattern = None
    for n in names:
        pred_dir = name_pth_map[n]
        if file_pattern is not None:
            for root, _, files in os.walk(pred_dir):
                for file in files:
                    if file_pattern.search(file):
                        final_files.append(os.path.join(root, file))
        if 'MSA' in options:
            msa_dir = os.path.join(os.path.dirname(pred_dir), 'msa')
            if not os.path.isdir(msa_dir):
                msa_dir = os.path.join(os.path.dirname(os.path.dirname(pred_dir)), 'msa')
            for file in os.listdir(msa_dir):
                if file.endswith('.csv'):
                    final_files.append(os.path.join(msa_dir, file))
    max_f_cnt_len = len(str(len(final_files)))
    yield f'{0:{max_f_cnt_len}}/{len(final_files)}', gr.update(), gr.update()
    c = 0
    with zipfile.ZipFile(zipped_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_f:
        for file in final_files:
            zip_f.write(file, os.path.relpath(file, output_dir))
            c += 1
            yield f'{c:{max_f_cnt_len}}/{len(final_files)}', gr.update(), gr.update()
    zipname_pth_map[os.path.basename(zipped_file)] = zipped_file
    yield 'Zipping done', zipped_file, zipname_pth_map

def remove_zip_file(gr_tmp_pth: str, zipname_pth_map: dict):
    # Remove the zip file to save disk space since gr.File already copy it to a new temp location
    basename = os.path.basename(gr_tmp_pth)
    os.remove(zipname_pth_map[basename])
    del zipname_pth_map[basename]
    return zipname_pth_map

def _extract_pred_dirs():
    name_path_map = {}
    for out_f_or_d in os.listdir(output_dir):
        rng_dir = os.path.join(output_dir, out_f_or_d)
        is_vhts = out_f_or_d.rsplit('_')[-2] == 'vHTS'
        if os.path.isdir(rng_dir):
            for target_pth in os.listdir(rng_dir):
                if 'boltz_results_' not in target_pth:
                    continue
                if not is_vhts:
                    target_dir = os.path.join(rng_dir, target_pth)
                    pred_parent_dir = os.path.join(target_dir, 'predictions')
                    if not os.path.isdir(pred_parent_dir):
                        shutil.rmtree(target_dir)
                        continue
                    for name in os.listdir(pred_parent_dir):
                        pred_dir = os.path.join(pred_parent_dir, name)
                        if not os.path.isdir(pred_dir):
                            continue
                        if name in name_path_map:
                            i = 2
                            new_name = f'{name}_{i}'
                            while new_name in name_path_map:
                                i += 1
                                new_name = f'{name}_{i}'
                        else:
                            new_name = name
                        name_path_map[new_name] = pred_dir
                else:
                    target_dir = os.path.join(rng_dir, target_pth)
                    pred_parent_dir = os.path.join(target_dir, 'predictions')
                    name = target_pth.split('boltz_results_', 1)[-1].rsplit('_', 1)[0]
                    if name in name_path_map:
                        i = 2
                        new_name = f'{name}_{i}'
                        while new_name in name_path_map:
                            i += 1
                            new_name = f'{name}_{i}'
                        name = new_name
                    name_path_map[name] = pred_parent_dir
    return name_path_map

def update_file_tree_and_dropdown():
    file_explorer = gr.FileExplorer(root_dir=output_dir,
                                    label='Output Files',
                                    interactive=True)
    name_path_map = _extract_pred_dirs()
    return file_explorer, gr.update(choices=list(name_path_map)), name_path_map

### Utilities ###
def rdkit_embed_molecule(lig):
    try:
        report = AllChem.EmbedMolecule(lig, useRandomCoords=True)
        if report == -1:
            return None
        else:
            return lig
    except Exception as e:
        return None

def rdkit_embed_with_timeout(lig, timeout):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(rdkit_embed_molecule, lig)
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            future.cancel()
            return None

def reverse_complementary_nucleic_acid(inp_na: str, type: str):
    if not inp_na.strip():
        return ''
    inp_na = inp_na.strip().upper()
    for i, c in enumerate(inp_na):
        if c not in 'ACTGU':
            return f'Invalid nucleic acid sequence! Position {i+1} is "{c}".'
    if type == 'Match Input':
        if 'U' in inp_na and 'T' in inp_na:
            return ('Both "U" and "T" are presented in input sequence!\n'
                    'Please manually specify which type of nucleic acid is required.')
        elif 'U' in inp_na:
            type = 'RNA'
        else:
            type = 'DNA'
    mapping_dict = rev_comp_map[type]
    return ''.join(mapping_dict[c] for c in inp_na[::-1])

def get_ligand_molstar_html(ccd_id: str):
    return f"""
    <iframe
        id="molstar_frame_ligand"
        style="width: 100%; height: 400px; border: none;"
        srcdoc='
            <!DOCTYPE html>
            <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
                </head>
                <body>
                    <div id="Viewer" style="width: 300px; height: 300px; position: center"></div>
                    <script>
                        (async function() {{
                            const viewer = new rcsbMolstar.LigandViewer("Viewer",
                            {{showWelcomeToast: false,
                              layoutShowControls: false}});
                            
                            const ccdID = "{ccd_id}";

                            try {{
                                await viewer.loadLigandId(ccdID);
                            }} catch (error) {{
                                console.error("Error loading structure:", error);
                            }}
                      }})();
                    </script>
                </body>
            </html>
        '>
    </iframe>"""

def get_mol_molstar_html(mol_str: str):
    mol_js_string = json.dumps(mol_str)
    return f"""
    <iframe
        style="width: 100%; height: 400px; border: none;"
        srcdoc='
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            </head>
            <body>
                <div id="Viewer" style="width: 100%; height: 100%;"></div>
                <script>
                    (async function() {{
                        const viewer = new rcsbMolstar.Viewer("Viewer", {{
                            showWelcomeToast: false,
                            layoutShowControls: false
                        }});
                        try {{
                            await viewer.loadStructureFromData({mol_js_string}, "mol", false);
                            viewer.plugin.managers.interactivity.setProps({{ granularity: "element" }});
                        }} catch (err) {{
                            console.error("Mol* load error:", err);
                        }}
                    }})();
                </script>
            </body>
            </html>
        '>
    </iframe>
    """

def draw_ccd_mol_3d(ccd_id: str):
    ccd_id = ccd_id.upper()
    yield get_ligand_molstar_html(ccd_id), pd.DataFrame()
    cif_url = f'https://files.rcsb.org/ligands/download/{ccd_id}.cif'
    result = requests.get(cif_url)
    if result.status_code == 404:
        yield get_ligand_molstar_html(''), pd.DataFrame()
    
    sio = io.StringIO(result.text)
    lig_dict = MMCIF2Dict(sio)
    chem_descriptor_prefix = '_pdbx_chem_comp_descriptor'
    looped_name = ['type', 'program', 'descriptor']
    data_dict = {}
    
    for k in lig_dict:
        if k.startswith(chem_descriptor_prefix):
            sub_k = k.rsplit('.', 1)[-1]
            if sub_k in looped_name:
                data_dict[sub_k.capitalize()] = [i.replace('"', '') for i in list(lig_dict[k])]
    
    yield gr.update(), pd.DataFrame(data_dict)

def draw_smiles_3d(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        yield get_mol_molstar_html(''), gr.update(value=pd.DataFrame({'Property': ['Error'],
                                                                      'Value': ['Invalid Molecule!']}))
    else:
        mol = Chem.AddHs(mol)
        data_dict = {'Property': list(property_functions), 'Value': []}
        for func in property_functions.values():
            v = func(mol)
            if isinstance(v, float):
                v = round(v, 4)
            data_dict['Value'].append(v)
        yield get_mol_molstar_html(''), gr.update(value=pd.DataFrame(data_dict))
        new_mol = rdkit_embed_with_timeout(mol, 60)
        if new_mol is None:
            mol = Chem.RemoveHs(mol)    # If embedding failed / timeout, just use 2D coord
        else:
            mol = Chem.RemoveHs(new_mol)
        mol_str = Chem.MolToMolBlock(mol)
        yield get_mol_molstar_html(mol_str), gr.update()

### Boltz Interface ###
with gr.Blocks(css=css, theme=gr.themes.Origin()) as Interface:
    gr.Markdown('<span style="font-size:25px; font-weight:bold;">Boltz Interface</span>')
    with gr.Tab('🧬 Single Complex'):
        gr.Markdown('<span style="font-size:20px; font-weight:bold;">Basic Settings</span>')
        template_name_chain_dict, template_name_path_dict,\
                template_name_usage_dict, template_name_setting_dict = \
                    gr.State({}), gr.State({}), gr.State({}), gr.State({})
        chain_res_dict = gr.State({})
        with gr.Accordion('Template', open=False):
            with gr.Row():
                with gr.Column():
                    template_file = gr.Files(label='mmCIF tempalte(s)', file_types=['.cif'],
                                                interactive=True)
                    template_dropdown = gr.Dropdown(label='Template Name', interactive=True)
                with gr.Column():
                    with gr.Row():
                        use_template_checkbox = gr.Checkbox(False, label='Use template',
                                                            interactive=False)
                        force_template_checkbox = gr.Checkbox(False, label='Force', interactive=False)
                    target_chain_ids   = gr.Dropdown(label='Target Chain IDs',
                                                        multiselect=True, interactive=True)
                    template_chain_ids = gr.Dropdown(label='Template Chain IDs',
                                                        multiselect=True, interactive=True)
                    template_threshold = gr.Number(None, label='Threshold (Å)', info='-1 = no threshold',
                                                   interactive=False, minimum=-1)
        
        with gr.Accordion('Constraints', open=False):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown('<span style="font-size:15px; font-weight:bold;">Bond conditioning</span>')
                    with gr.Group():
                        with gr.Row():
                            with gr.Column(min_width=60):
                                atom1_chain_dropdown = gr.Dropdown(label='Atom1 Chain',
                                                                interactive=True)
                                atom1_res_dropdown   = gr.Dropdown(label='Atom1 Residue',
                                                                interactive=True)
                                atom1_atmname_text   = gr.Text(label='Atom1 Name',
                                                            interactive=True)
                            with gr.Column(min_width=60):
                                atom2_chain_dropdown = gr.Dropdown(label='Atom2 Chain',
                                                                interactive=True)
                                atom2_res_dropdown   = gr.Dropdown(label='Atom2 Residue',
                                                                interactive=True)
                                atom2_atmname_text   = gr.Text(label='Atom2 Name',
                                                            interactive=True)
                    atom1_chain_dropdown.change(update_bond_sequence_length_with_chain,
                                                inputs=[atom1_chain_dropdown, chain_res_dict],
                                                outputs=atom1_res_dropdown)
                    atom2_chain_dropdown.change(update_bond_sequence_length_with_chain,
                                                inputs=[atom2_chain_dropdown, chain_res_dict],
                                                outputs=atom2_res_dropdown)
                
                with gr.Column(scale=1):
                    gr.Markdown('<span style="font-size:15px; font-weight:bold;">Pocket conditioning</span>')
                    with gr.Group():
                        pocket_binder = gr.Dropdown(label='Binder',
                                                    interactive=True)
                        pocket_text = gr.Text(label='Target Pockets',
                                            placeholder='B:12,B:23',
                                            interactive=True)
                        with gr.Row():
                            pocket_max_distance = gr.Number(6, label='Max Distance (Å)',
                                                            interactive=True, minimum=0,
                                                            min_width=20, scale=2)
                            pocket_force_checkbox = gr.Checkbox(False, label='Force', min_width=20,
                                                                scale=1, elem_classes='centered-checkbox')
                
                with gr.Column(scale=2):
                    gr.Markdown('<span style="font-size:15px; font-weight:bold;">Contact Conditioning</span>')
                    with gr.Group():
                        with gr.Row():
                            contact_1_dropdown = gr.Dropdown(label='Chain 1',
                                                            interactive=True)
                            contact_1_text = gr.Text(label='Residue IDX/Atom Name')
                        with gr.Row():
                            contact_2_dropdown = gr.Dropdown(label='Chain 2',
                                                            interactive=True)
                            contact_2_text = gr.Text(label='Residue IDX/Atom Name')
                        with gr.Row():
                            contact_max_distance = gr.Number(6, label='Max Distance (Å)',
                                                             interactive=True, minimum=0, min_width=20, scale=2)
                            contact_force_checkbox = gr.Checkbox(False, label='Force', min_width=20, scale=1,
                                                                 elem_classes='centered-checkbox')
        
        with gr.Row():
            with gr.Column():
                gr.Markdown('<span style="font-size:15px; font-weight:bold;">Name, Affinity & Entities</span>')
                single_complex_name = gr.Text(label='Name',
                                              placeholder='Complex_1',
                                              interactive=True)
                affinity_binder = gr.Dropdown(label='Affinity Prediction Chain',
                                              interactive=True)
                mod_entity_number = gr.Number(1, label='Total Entity',
                                              interactive=True, minimum=1, step=1)
        
        
        def update_all_chains_dropdown(*all_entity_chain_values):
            all_chains = set()
            affinity_chains = set()
            for i in range(0, len(all_entity_chain_values), 2):
                entity, chain = all_entity_chain_values[i:i+2]
                chains = [c.strip() for c in chain.split(',') if c.strip()]
                all_chains.update(chains)
                if entity in ['Ligand', 'CCD']:
                    affinity_chains.update(chains)
            final_choices = [''] + sorted(all_chains)
            aff_final = [''] + sorted(affinity_chains)
            return (gr.update(choices=final_choices), gr.update(choices=aff_final),
                    gr.update(choices=final_choices), gr.update(choices=final_choices),
                    gr.update(choices=final_choices),)
        
        @gr.render(inputs=mod_entity_number)
        def append_new_entity(counts: int):
            component_cnt = 7
            component_refs = []
            for i in range(counts):
                gr.Markdown(f'<span style="font-size:15px; font-weight:bold;">Entity {i+1}</span>', key=f'MK_{i}')
                with gr.Row(key=f'Entity_{i}'):
                    with gr.Column(key=f'Entity_{i}_sub1', scale=1):
                        entity_menu = gr.Dropdown(entity_types,
                                                label='Entity',
                                                value=entity_types[0],
                                                interactive=True,
                                                key=f'ET_{i}', scale=1)
                        chain_name_text = gr.Text('',
                                                label='Chains',
                                                info='Press Enter to update Chains',
                                                placeholder='A,B,C',
                                                interactive=True,
                                                key=f'Chain_{i}',
                                                scale=1)
                    with gr.Column(key=f'Entity_{i}_sub2', scale=5):
                        with gr.Group(key=f'Entity_{i}_sub2_Group'):
                            sequence_text = gr.TextArea(label='Sequence',
                                                        placeholder='Input',
                                                        interactive=True,
                                                        lines=3,
                                                        key=f'SQ_{i}',
                                                        elem_classes='sequence')
                            highlight_text = gr.HighlightedText([('Input required!', 'X')],
                                                                label='Validation',
                                                                color_map={'✓': 'green',
                                                                        'X': 'red'},
                                                                key=f'HL_{i}',
                                                                elem_classes='validation',
                                                                show_legend=True)
                    with gr.Column(key=f'Entity_{i}_sub3', scale=1):
                        with gr.Group(key=f'Entity_{i}_sub3_group1'):
                            with gr.Row(key=f'Entity_{i}_sub3_group1_row1'):
                                cyclic_ckbox = gr.Checkbox(False, label='Cyclic', min_width=50)
                                msa_ckbox = gr.Checkbox(True, label='Use MSA', min_width=50, interactive=True)
                            modification_text = gr.Text(label='Modifications (Residue:CCD)',
                                                        placeholder='2:ALY,15:MSE')
                            msa_file = gr.File(label='MSA File', file_types=['.a3m', '.csv'], height=92,
                                               elem_classes='small-upload-style')
                        
                    component_refs.extend([entity_menu, chain_name_text, sequence_text,
                                           cyclic_ckbox, modification_text, msa_file, msa_ckbox])
                    entity_menu.change(change_sequence_label,
                                       inputs=[entity_menu, sequence_text, cyclic_ckbox],
                                       outputs=[sequence_text, highlight_text, cyclic_ckbox])
                    sequence_text.change(validate_sequence,
                                        inputs=[entity_menu, sequence_text],
                                        outputs=[highlight_text])
                    chain_name_text.submit(update_chain_seq_dict,
                                           inputs=[entity_menu, chain_name_text,
                                                   sequence_text, chain_res_dict],
                                           outputs=[chain_res_dict, atom1_chain_dropdown, atom2_chain_dropdown])
                    chain_name_text.input(update_chain_seq_dict,
                                          inputs=[entity_menu, chain_name_text,
                                                  sequence_text, chain_res_dict],
                                          outputs=[chain_res_dict, atom1_chain_dropdown, atom2_chain_dropdown])
                    entity_menu.change(update_chain_seq_dict,
                                       inputs=[entity_menu, chain_name_text,
                                               sequence_text, chain_res_dict],
                                       outputs=[chain_res_dict, atom1_chain_dropdown, atom2_chain_dropdown])
                
                gr.HTML("<hr>")
            
            chain_components = [comp for i, comp in enumerate(component_refs) if i % component_cnt <= 1]
            entity_components = [comp for i, comp in enumerate(component_refs) if i % component_cnt == 0]
            for i in range(0, len(chain_components), 2):
                chain_input = chain_components[i+1]
                entity_menu = entity_components[i//2]
                chain_input.submit(update_all_chains_dropdown,
                                   inputs=chain_components,
                                   outputs=[pocket_binder, affinity_binder,
                                            contact_1_dropdown, contact_2_dropdown,
                                            target_chain_ids])
                chain_input.input(update_all_chains_dropdown,
                                  inputs=chain_components,
                                  outputs=[pocket_binder, affinity_binder,
                                           contact_1_dropdown, contact_2_dropdown,
                                           target_chain_ids])
                entity_menu.change(update_all_chains_dropdown,
                                   inputs=chain_components,
                                   outputs=[pocket_binder, affinity_binder,
                                            contact_1_dropdown, contact_2_dropdown,
                                            target_chain_ids])
            
            def write_yaml_func(binder, target, pocket_max_d, pocket_f, aff_binder,
                                cont_1_c, cont_1_r, cont_2_c, cont_2_r, contact_max_dist, contact_f,
                                template_name_path_dict, template_name_usage_dict,
                                template_name_setting_dict,
                                bond_atom1_chain, bond_atom1_res, bond_atom1_name,
                                bond_atom2_chain, bond_atom2_res, bond_atom2_name,
                                *all_components):
                all_components = list(all_components)
                # TODO: Add more advanced format validation functions!
                
                # constraints --> pocket
                if binder and target:
                    contacts = []
                    for c_res in target.split(','):
                        if ':' not in c_res:
                            return ('Invalid target pocket, please use ":" to '
                                    'separate target chain and target residue!\n'
                                    'E.g., B:12,C:13')
                        c, r = c_res.split(':')
                        contacts.append([c, int(r)])
                    d = {'pocket': {'binder'      : binder,
                                    'contacts'    : contacts,
                                    'force'       : pocket_f}}
                    if pocket_max_d:
                        d['pocket']['max_distance'] = pocket_max_d
                    data_dict = {'sequences': [],
                                 'constraints': [d]}
                else:
                    data_dict = {'sequences': []}
                
                # constraints --> contact
                if cont_1_c and cont_1_r.strip() and cont_2_c and cont_2_r.strip():
                    cont_1_r = cont_1_r.strip()
                    cont_2_r = cont_2_r.strip()
                    if cont_1_r.isdigit():
                        cont_1_r = int(cont_1_r)
                    if cont_2_r.isdigit():
                        cont_2_r = int(cont_2_r)
                    contact_dict = {'contact': {'token1': [cont_1_c, cont_1_r],
                                                'token2': [cont_2_c, cont_2_r],
                                                'force'       : contact_f}}
                    if contact_max_dist:
                        contact_dict['contact']['max_distance'] = contact_max_dist
                    if 'constraints' in data_dict:
                        data_dict['constraints'].append(contact_dict)
                    else:
                        data_dict['constraints'] = [contact_dict]
                
                # constraints --> bond
                if all((bond_atom1_chain, bond_atom1_res, bond_atom1_name,
                        bond_atom2_chain, bond_atom2_res, bond_atom2_name)):
                    bond_dict = {'bond': {'atom1': [bond_atom1_chain, bond_atom1_res, bond_atom1_name.strip()],
                                          'atom2': [bond_atom2_chain, bond_atom2_res, bond_atom2_name.strip()],}}
                    if 'constraints' in data_dict:
                        data_dict['constraints'].append(bond_dict)
                    else:
                        data_dict['constraints'] = [bond_dict]
                
                # properties
                if aff_binder:
                    data_dict.update({'properties': [{'affinity': {'binder': aff_binder}}]})
                
                # templates
                all_templates = []
                for name in template_name_path_dict:
                    if template_name_usage_dict[name]:
                        curr_template = {'cif': template_name_path_dict[name]}
                        chain_template_id_dict = template_name_setting_dict[name]
                        curr_template['force'] = chain_template_id_dict['force']
                        if chain_template_id_dict['threshold'] > -1:
                            curr_template['threshold'] = chain_template_id_dict['threshold']
                        else:
                            if curr_template['force']:
                                return ('Template threshold must be specified if "Force" is set to true!')
                        if chain_template_id_dict['chain_id']:
                            curr_template['chain_id'] = chain_template_id_dict['chain_id']
                        if chain_template_id_dict['template_id']:
                            curr_template['template_id'] = chain_template_id_dict['template_id']
                        all_templates.append(curr_template)
                if all_templates:
                    data_dict.update({'templates': all_templates})
                
                existing_chains = []
                msa_rng_name = uuid.uuid4().hex[:8]
                
                for i in range(0, len(all_components), component_cnt):
                    entity, chain, seq, cyclic, mod, msa_pth, use_msa = all_components[i:i+component_cnt]
                    seq = seq.strip()
                    
                    # set entity type
                    if entity == 'CCD':
                        entity_type = 'ligand'
                    else:
                        entity_type = entity.lower()
                    
                    # set chain id
                    chains = chain.split(',')
                    if len(chains) == 1:
                        id = chain.strip()
                        if id in existing_chains:
                            return f'Chain {id} of Entity {i//6+1} already existed!'
                        existing_chains.append(id)
                    else:
                        id = [c.strip() for c in chains]
                        for _i in id:
                            if id.count(_i) > 1:
                                return f'Duplicate chain found within Entity {i//6+1}!'
                            if _i in existing_chains:
                                return f'Chain {id} of Entity {i//6+1} already existed!'
                        existing_chains.extend(id)
                    
                    # set key of sequence ('sequence', 'ccd' or 'smiles')
                    if not seq:
                        return f'Entity {i//5+1} is empty!'
                    if entity == 'CCD':
                        seq_key = 'ccd'
                        seq = seq.upper()
                        if not re.fullmatch(r'(?:[A-Z0-9]{3}|[A-Z0-9]{5})|[A-Z]{2}', seq):
                            return f'Entity {i//5+1} is not a valid CCD ID!'
                    elif entity == 'Ligand':
                        seq_key = 'smiles'
                        if Chem.MolFromSmiles(seq) is None:
                            return f'Entity {i//5+1} is not a valid SMILES!'
                    else:
                        seq = seq.upper()
                        seq_key = 'sequence'
                        valid_strs = allow_char_dict[entity]
                        for char in seq:
                            if char not in valid_strs:
                                return f'Entity {i//5+1} is not a valid {entity}!'
                    
                    # set modification
                    if mod:
                        modifications = []
                        all_mods = mod.split(',')
                        for pos_ccd in all_mods:
                            if ':' not in pos_ccd:
                                return (f'Invalid modification for Entity {i//6+1}, please use ":" to '
                                        f'separate residue and CCD!\n')
                            pos, ccd = pos_ccd.split(':')
                            modifications.append({'position': int(pos), 'ccd': ccd})
                    else:
                        modifications = None
                    
                    if entity_type == 'ligand':
                        curr_dict = {entity_type: {'id'    : id,
                                                   seq_key : seq,}}
                    else:
                        curr_dict = {entity_type: {'id'    : id,
                                                   seq_key : seq.upper(),
                                                   'cyclic': cyclic}}
                    if entity_type == 'protein':
                        if msa_pth and use_msa:
                            target_msa = os.path.join(msa_dir, msa_rng_name, os.path.basename(msa_pth))
                            os.makedirs(os.path.dirname(target_msa), exist_ok=True)
                            shutil.copy(msa_pth, target_msa)
                            curr_dict[entity_type]['msa'] = target_msa
                        elif not use_msa:
                            curr_dict[entity_type]['msa'] = 'empty'
                    if modifications is not None:
                        curr_dict[entity_type]['modifications'] = modifications
                    
                    data_dict['sequences'].append(curr_dict)
                
                yaml_string = safe_dump(data_dict, sort_keys=False, indent=4)
                return yaml_string
                
            write_yaml_button.click(write_yaml_func,
                                    inputs=[pocket_binder, pocket_text,
                                            pocket_max_distance, pocket_force_checkbox, affinity_binder,
                                            contact_1_dropdown, contact_1_text,
                                            contact_2_dropdown, contact_2_text,
                                            contact_force_checkbox, contact_max_distance,
                                            template_name_path_dict,
                                            template_name_usage_dict, 
                                            template_name_setting_dict,
                                            atom1_chain_dropdown, atom1_res_dropdown, atom1_atmname_text,
                                            atom2_chain_dropdown, atom2_res_dropdown, atom2_atmname_text,
                                            *component_refs],
                                    outputs=[yaml_text])
        
        with gr.Row():
            with gr.Column():
                write_yaml_button = gr.Button('Write YAML')
                add_single_to_bacth_button = gr.Button('Add to Batch')
                run_single_boltz_button = gr.Button('Run Boltz', interactive=False)
            yaml_text = gr.Code(label='YAML Output',
                                scale=4,
                                language='yaml',
                                interactive=True, max_lines=15)
        
        single_boltz_log = gr.Textbox(label='Prediction Log', lines=10, max_lines=10,
                                      autofocus=False, elem_classes='log', show_copy_button=True)
        
        template_file.upload(read_tempaltes,
                             inputs=[template_file,
                                     template_name_chain_dict, template_name_path_dict,
                                     template_name_usage_dict, template_name_setting_dict],
                             outputs=[template_dropdown,
                                      template_name_chain_dict, template_name_path_dict,
                                      template_name_usage_dict, use_template_checkbox,
                                      template_name_setting_dict, force_template_checkbox, template_threshold])
        use_template_checkbox.input(update_template_chain_ids_and_settings,
                                    inputs=[use_template_checkbox, target_chain_ids, template_chain_ids,
                                            force_template_checkbox, template_threshold,
                                            template_dropdown, template_name_usage_dict, template_name_setting_dict],
                                    outputs=[template_name_usage_dict, template_name_setting_dict])
        target_chain_ids.input(update_template_chain_ids_and_settings,
                               inputs=[use_template_checkbox, target_chain_ids, template_chain_ids,
                                       force_template_checkbox, template_threshold,
                                       template_dropdown, template_name_usage_dict, template_name_setting_dict],
                               outputs=[template_name_usage_dict, template_name_setting_dict])
        template_chain_ids.input(update_template_chain_ids_and_settings,
                                 inputs=[use_template_checkbox, target_chain_ids, template_chain_ids,
                                         force_template_checkbox, template_threshold,
                                         template_dropdown, template_name_usage_dict, template_name_setting_dict],
                                 outputs=[template_name_usage_dict, template_name_setting_dict])
        force_template_checkbox.input(update_template_chain_ids_and_settings,
                                 inputs=[use_template_checkbox, target_chain_ids, template_chain_ids,
                                         force_template_checkbox, template_threshold,
                                         template_dropdown, template_name_usage_dict, template_name_setting_dict],
                                 outputs=[template_name_usage_dict, template_name_setting_dict])
        template_threshold.change(update_template_chain_ids_and_settings,
                                 inputs=[use_template_checkbox, target_chain_ids, template_chain_ids,
                                         force_template_checkbox, template_threshold,
                                         template_dropdown, template_name_usage_dict, template_name_setting_dict],
                                 outputs=[template_name_usage_dict, template_name_setting_dict])
        template_dropdown.change(update_template_dropdown,
                                 inputs=[template_dropdown, template_name_chain_dict,
                                         template_name_usage_dict, template_name_setting_dict],
                                 outputs=[use_template_checkbox, target_chain_ids, template_chain_ids,
                                          force_template_checkbox, template_threshold])
        
        single_complex_name.input(check_yaml_strings,
                                  inputs=[yaml_text, single_complex_name],
                                  outputs=run_single_boltz_button)
        yaml_text.change(check_yaml_strings,
                         inputs=[yaml_text, single_complex_name],
                         outputs=run_single_boltz_button)
        run_single_boltz_button.click(execute_single_boltz,
                                       inputs=[single_complex_name, yaml_text,
                                               *all_boltz_parameters],
                                       outputs=[run_single_boltz_button, single_boltz_log])
    
    
    with gr.Tab('🚀 Batch Predict'):
        batch_upload_files = gr.State({})
        processed_inp_files = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                mod_batch_total_files = gr.Number(0, label='Total Files',
                                                  scale=1, interactive=True,
                                                  minimum=0, step=1)
                clear_batch_button = gr.Button('Clear')
            upload_yaml_files = gr.Files(file_types=['.yaml', '.yml'],
                                         label='Upload YAML files',
                                         interactive=True, scale=2)
        
        upload_yaml_files.upload(upload_multi_files,
                                 inputs=[upload_yaml_files, mod_batch_total_files],
                                 outputs=[batch_upload_files, mod_batch_total_files, upload_yaml_files])
        add_single_to_bacth_button.click(add_current_single_to_batch,
                                         inputs=[single_complex_name, yaml_text,
                                                 batch_upload_files, mod_batch_total_files],
                                         outputs=[batch_upload_files, mod_batch_total_files,
                                                  add_single_to_bacth_button])
        clear_batch_button.click(clear_curr_batch_dict,
                                 outputs=[batch_upload_files, mod_batch_total_files])
        
        @gr.render(inputs=[batch_upload_files, mod_batch_total_files],
                   triggers=[clear_batch_button.click, mod_batch_total_files.change])
        def create_new_batch_file_count(all_uploaded_files: dict, counts: int):
            batched_files = []
            total_uploaded = len(all_uploaded_files)
            paired_all_files = list(all_uploaded_files.items())
            pair_c = 0
            for i in range(counts):
                gr.Markdown(f'<span style="font-size:15px; font-weight:bold;">File {i+1}</span>', key=f'B_MK_{i}')
                with gr.Row(key=f'B_File_{i}'):
                    if i >= counts - total_uploaded:
                        name_str = paired_all_files[min(pair_c, counts-1)][0]
                        yaml_str = paired_all_files[min(pair_c, counts-1)][1]
                        file_name_text = gr.Text(name_str,
                                                 label='Name',
                                                 interactive=True, scale=1,
                                                 key=f'name_{i}')
                        yaml_str_code = gr.Code(yaml_str,
                                                label='YAML String',
                                                language='yaml',
                                                interactive=True, scale=4, max_lines=10,
                                                key=f'yaml_{i}')
                        pair_c += 1
                    else:
                        file_name_text = gr.Text(label='Name',
                                                 interactive=True, scale=1,
                                                 key=f'name_{i}')
                        yaml_str_code = gr.Code(label='YAML String',
                                                language='yaml',
                                                interactive=True, scale=4, max_lines=10,
                                                key=f'yaml_{i}')
                batched_files.extend([file_name_text, yaml_str_code])
                file_name_text.input(check_batch_yaml_and_name,
                                     inputs=[yaml_str_code, file_name_text],
                                     outputs=file_name_text)
                yaml_str_code.input(check_batch_yaml_and_name,
                                    inputs=[yaml_str_code, file_name_text],
                                    outputs=file_name_text)
                
                gr.HTML("<hr>")
            
            def process_all_files(*all_components):
                all_components = list(all_components)
                final_result_map = {}
                inp_rng_dir = os.path.join(input_dir, f"batch_{uuid.uuid4().hex[:8]}")
                check_dir_exist_and_rename(inp_rng_dir)
                for i in range(0, len(all_components), 2):
                    file_name, yaml_str = all_components[i:i+2]
                    file_name = file_name.strip()
                    yaml_valid = _check_yaml_strings(yaml_str)
                    if file_name and yaml_valid:
                        final_result_map[os.path.join(inp_rng_dir, f'{file_name}.yaml')] = yaml_str
                for f_pth, yaml_content in final_result_map.items():
                    with open(f_pth, 'w') as f:
                        f.write(yaml_content)
                return list(final_result_map), gr.update(interactive=True), list(final_result_map)
            
            batch_process_all_files.click(process_all_files,
                                          inputs=batched_files,
                                          outputs=[batch_process_result,
                                                   batch_predict_all_files,
                                                   processed_inp_files])
        
        with gr.Row():
            with gr.Column(scale=1):
                batch_process_all_files = gr.Button('Batch Process')
                batch_predict_all_files = gr.Button('Batch Predict', interactive=False)
            with gr.Column(scale=2):
                batch_process_result = gr.File(label='Processed Files',
                                               interactive=False, file_count='multiple')
        
        multi_boltz_log = gr.Textbox(label='Prediction Log', lines=10, max_lines=10,
                                     autofocus=False, elem_classes='log', show_copy_button=True)
        
        batch_predict_all_files.click(execute_multi_boltz,
                                      inputs=[processed_inp_files,
                                              *all_boltz_parameters],
                                      outputs=[batch_predict_all_files, multi_boltz_log])
    
    
    with gr.Tab('🧪 vHTS'):
        gr.Markdown('<span style="font-size:20px; font-weight:bold;">Multiple molecules vs Single protein</span>')
        with gr.Accordion('1. Ligand Settings', open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        ligand_file_type = gr.Dropdown(['Chemical files', 'Tabular files'],
                                                       'Chemical files',
                                                       label='Chemical format type',
                                                       interactive=True)
                        vhts_ligand_chain_text = gr.Text(label='Ligand Chain', placeholder='Z',
                                                         interactive=True, value='Z')
                    vhts_clear_ligand_df_btn = gr.Button('Clear Ligands')
                    ligand_dataframe = gr.DataFrame(headers=['Name', 'SMILES'],
                                                    max_height=300, interactive=True, min_width=250,
                                                    show_row_numbers=True)
                with gr.Column():
                    chemical_file_upload_file = gr.File(label='Upload chemical file(s)', interactive=True,
                                                        file_count='multiple',
                                                        file_types=['.sdf', '.mol', '.smi', '.zip'])
                    with gr.Row():
                        tabular_chem_file_name_id = gr.Text(label='Column of Name',
                                                            interactive=True, visible=False)
                        tabular_chem_file_chem_id = gr.Text(label='Column of Chem String',
                                                            interactive=True, visible=False)
                        tabular_chem_file_delimiter = gr.Dropdown([',', '\t', ';', ' '], value=',',
                                                                  label='Delimiter',
                                                                  allow_custom_value=True, visible=False,
                                                                  interactive=True)
            ligand_file_type.input(update_chem_file_format, inputs=ligand_file_type,
                                   outputs=[chemical_file_upload_file, tabular_chem_file_name_id,
                                            tabular_chem_file_chem_id, tabular_chem_file_delimiter])
            vhts_clear_ligand_df_btn.click(lambda x: pd.DataFrame(), inputs=ligand_dataframe, outputs=ligand_dataframe)
            chemical_file_upload_file.upload(process_uploaded_ligand,
                                             inputs=[chemical_file_upload_file, tabular_chem_file_name_id,
                                                     tabular_chem_file_chem_id, tabular_chem_file_delimiter,
                                                     ligand_dataframe],
                                             outputs=[ligand_dataframe])
        
        with gr.Accordion('2. Protein Settings', open=False):
            with gr.Accordion('Template', open=False):
                with gr.Row():
                    vhts_template_name_chain_dict, vhts_template_name_path_dict,\
                        vhts_template_name_usage_dict, vhts_template_name_setting_dict = \
                            gr.State({}), gr.State({}), gr.State({}), gr.State({})
                    with gr.Group():
                        vhts_template_file = gr.Files(label='mmCIF tempalte(s)', file_types=['.cif'],
                                                    interactive=True)
                        vhts_template_dropdown = gr.Dropdown(label='Template Name', interactive=True)
                    with gr.Group():
                        with gr.Row():
                            vhts_use_template_chekcbox = gr.Checkbox(False, label='Use template',
                                                                        interactive=False)
                            vhts_force_template_checkbox = gr.Checkbox(False, label='Force', interactive=False)
                        vhts_target_chain_ids   = gr.Dropdown(label='Target Chain IDs',
                                                                multiselect=True, interactive=True)
                        vhts_template_chain_ids = gr.Dropdown(label='Template Chain IDs',
                                                                multiselect=True, interactive=True)
                        vhts_template_threshold = gr.Number(None, label='Threshold (Å)', info='-1 = no threshold',
                                                            interactive=False, minimum=-1)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<span style="font-size:15px; font-weight:bold;">Pocket Conditioning & Entity Count</span>')
                    with gr.Group():
                        vhts_pocket_text = gr.Text(label='Target Pockets',
                                                placeholder='B:12,B:23',
                                                interactive=True)
                        with gr.Row():
                            vhts_pocket_max_distance = gr.Number(6, label='Max Distance (Å)',
                                                                interactive=True, minimum=0,
                                                                min_width=20, scale=2)
                            vhts_pocket_force_checkbox = gr.Checkbox(False, label='Force', min_width=20,
                                                                     scale=1, elem_classes='centered-checkbox')
                        vhts_entity_number = gr.Number(1, label='Total Entity',
                                                    interactive=True, minimum=1, step=1)
                with gr.Column(scale=1):
                    gr.Markdown('<span style="font-size:15px; font-weight:bold;">Contact Conditioning</span>')
                    with gr.Group():
                        with gr.Row():
                            vhts_contact_1_dropdown = gr.Dropdown(label='Chain 1',
                                                                  interactive=True)
                            vhts_contact_1_text = gr.Text(label='Residue')
                        with gr.Row():
                            vhts_contact_2_dropdown = gr.Dropdown(label='Chain 2',
                                                                  interactive=True)
                            vhts_contact_2_text = gr.Text(label='Residue')
                        with gr.Row():
                            vhts_contact_max_distance = gr.Number(6, label='Max Distance (Å)',
                                                                interactive=True, minimum=0, min_width=20, scale=2)
                            vhts_contact_force_checkbox = gr.Checkbox(False, label='Force', min_width=20,
                                                                      scale=1, elem_classes='centered-checkbox')
            
            def vhts_update_all_chains_dropdown(*all_entity_chain_values):
                all_chains = set()
                for chain in all_entity_chain_values:
                    chains = [c.strip() for c in chain.split(',') if c.strip()]
                    all_chains.update(chains)
                final_choices = [''] + sorted(all_chains)
                return (gr.update(choices=final_choices), gr.update(choices=final_choices),
                        gr.update(choices=final_choices))
            
            @gr.render(inputs=vhts_entity_number)
            def vhts_append_new_entity(counts: int):
                component_refs = []
                for i in range(counts):
                    gr.Markdown(f'<span style="font-size:15px; font-weight:bold;">Entity {i+1}</span>', key=f'MK_{i}')
                    with gr.Row(key=f'Entity_{i}'):
                        with gr.Column(key=f'Entity_{i}_sub1', scale=1):
                            entity_menu = gr.Dropdown(entity_types,
                                                      label='Entity',
                                                      value=entity_types[0],
                                                      interactive=True,
                                                      key=f'ET_{i}', scale=1)
                            chain_name_text = gr.Text('',
                                                      label='Chains',
                                                      info='Press Enter to update Binder',
                                                      placeholder='A,B,C',
                                                      interactive=True,
                                                      key=f'Chain_{i}',
                                                      scale=1)
                        with gr.Column(key=f'Entity_{i}_sub2', scale=4):
                            with gr.Group(key=f'Entity_{i}_sub2_Group'):
                                sequence_text = gr.TextArea(label='Sequence',
                                                            placeholder='Input',
                                                            interactive=True,
                                                            lines=3,
                                                            key=f'SQ_{i}',
                                                            elem_classes='sequence')
                                highlight_text = gr.HighlightedText([('Input required!', 'X')],
                                                                    label='Validation',
                                                                    color_map={'✓': 'green',
                                                                               'X': 'red'},
                                                                    key=f'HL_{i}',
                                                                    elem_classes='validation',
                                                                    show_legend=True)
                        with gr.Column(key=f'Entity_{i}_sub3', scale=1):
                            cyclic_ckbox = gr.Checkbox(False, label='Cyclic')
                            modification_text = gr.Text(label='Modifications (Residue:CCD)',
                                                        placeholder='2:ALY,15:MSE')
                        component_refs.extend([entity_menu, chain_name_text, sequence_text,
                                               cyclic_ckbox, modification_text])
                        entity_menu.change(change_sequence_label,
                                           inputs=[entity_menu, sequence_text, cyclic_ckbox],
                                           outputs=[sequence_text, highlight_text, cyclic_ckbox])
                        sequence_text.change(validate_sequence,
                                             inputs=[entity_menu, sequence_text],
                                             outputs=highlight_text)
                
                    gr.HTML("<hr>")
                chain_components = [comp for i, comp in enumerate(component_refs) if i % 5 == 1]
                entity_components = [comp for i, comp in enumerate(component_refs) if i % 5 == 0]
                for i, chain_input in enumerate(chain_components):
                    chain_input.submit(vhts_update_all_chains_dropdown,
                                       inputs=chain_components,
                                       outputs=[vhts_contact_1_dropdown, vhts_contact_2_dropdown,
                                                vhts_target_chain_ids])
                    entity_components[i].change(vhts_update_all_chains_dropdown,
                                                inputs=chain_components,
                                                outputs=[vhts_contact_1_dropdown, vhts_contact_2_dropdown,
                                                         vhts_target_chain_ids])
                
                def write_yaml_func(binder, target, pocket_max_d, pocket_f, aff_binder,
                                    cont_1_c, cont_1_r, cont_2_c, cont_2_r, contact_max_dist, contact_f,
                                    template_name_path_dict, template_name_usage_dict,
                                    template_name_setting_dict,
                                    *all_components):
                    all_components = list(all_components)
                    if not binder:
                        return 'Ligand chain must be provided!'
                    
                    if binder and target:
                        contacts = []
                        for c_res in target.split(','):
                            if ':' not in c_res:
                                return ('Invalid target pocket, please use ":" to '
                                        'separate target chain and target residue!\n'
                                        'E.g., B:12,C:13')
                            c, r = c_res.split(':')
                            contacts.append([c, int(r)])
                        d = {'pocket': {'binder'      : binder,
                                        'contacts'    : contacts,
                                        'force'       : pocket_f}}
                        if pocket_max_d:
                            d['pocket']['max_distance'] = pocket_max_d
                        data_dict = {'sequences': [],
                                     'constraints': [d]}
                    else:
                        data_dict = {'sequences': []}
                    if aff_binder:
                        data_dict.update({'properties': [{'affinity': {'binder': aff_binder}}]})
                    
                    if cont_1_c and cont_1_r.strip() and cont_2_c and cont_2_r.strip():
                        cont_1_r = cont_1_r.strip()
                        cont_2_r = cont_2_r.strip()
                        if cont_1_r.isdigit():
                            cont_1_r = int(cont_1_r)
                        if cont_2_r.isdigit():
                            cont_2_r = int(cont_2_r)
                        contact_dict = {'contact': {'token1': [cont_1_c, cont_1_r],
                                                    'token2': [cont_2_c, cont_2_r],
                                                    'force' : contact_f}}
                        if contact_max_dist:
                            contact_dict['contact']['max_distance'] = contact_max_dist
                        if 'constraints' in data_dict:
                            data_dict['constraints'].append(contact_dict)
                        else:
                            data_dict['constraints'] = [contact_dict]
                    
                    all_templates = []
                    for name in template_name_path_dict:
                        if template_name_usage_dict[name]:
                            curr_template = {'cif': template_name_path_dict[name]}
                            chain_template_id_dict = template_name_setting_dict[name]
                            curr_template['force'] = chain_template_id_dict['force']
                            if chain_template_id_dict['threshold'] > -1:
                                curr_template['threshold'] = chain_template_id_dict['threshold']
                            else:
                                if curr_template['force']:
                                    return ('Template threshold must be specified if "Force" is set to true!')
                            if chain_template_id_dict['chain_id']:
                                curr_template['chain_id'] = chain_template_id_dict['chain_id']
                            if chain_template_id_dict['template_id']:
                                curr_template['template_id'] = chain_template_id_dict['template_id']
                            all_templates.append(curr_template)
                    if all_templates:
                        data_dict.update({'templates': all_templates})
                    
                    existing_chains = []
                    
                    all_components += ['Ligand', binder, 'c1ccccc1', False, '']
                    
                    for i in range(0, len(all_components), 5):
                        entity, chain, seq, cyclic, mod = all_components[i:i+5]
                        seq = seq.strip()
                        
                        # set entity type
                        if entity == 'CCD':
                            entity_type = 'ligand'
                        else:
                            entity_type = entity.lower()
                        
                        # set chain id
                        chains = chain.split(',')
                        if len(chains) == 1:
                            id = chain.strip()
                            if id in existing_chains:
                                return f'Chain {id} of Entity {i//5+1} already existed!'
                            existing_chains.append(id)
                        else:
                            id = [c.strip() for c in chains]
                            for _i in id:
                                if id.count(_i) > 1:
                                    return f'Duplicate chain found within Entity {i//5+1}!'
                                if _i in existing_chains:
                                    return f'Chain {id} of Entity {i//5+1} already existed!'
                            existing_chains.extend(id)
                        
                        # set key of sequence ('sequence', 'ccd' or 'smiles')
                        if not seq:
                            return f'Entity {i//5+1} is empty!'
                        if entity == 'CCD':
                            seq = seq.upper()
                            seq_key = 'ccd'
                            if not re.fullmatch(r'(?:[A-Z0-9]{3}|[A-Z0-9]{5}|[A-Z]{2})', seq):
                                return f'Entity {i//5+1} is not a valid CCD ID!'
                        elif entity == 'Ligand':
                            seq_key = 'smiles'
                            if Chem.MolFromSmiles(seq) is None:
                                return f'Entity {i//5+1} is not a valid SMILES!'
                        else:
                            seq = seq.upper()
                            seq_key = 'sequence'
                            valid_strs = allow_char_dict[entity]
                            for char in seq:
                                if char not in valid_strs:
                                    return f'Entity {i//5+1} is not a valid {entity}!'
                        
                        # set modification
                        if mod:
                            modifications = []
                            all_mods = mod.split(',')
                            for pos_ccd in all_mods:
                                if ':' not in pos_ccd:
                                    return (f'Invalid modification for Entity {i//5+1}, please use ":" to '
                                            f'separate residue and CCD!\n')
                                pos, ccd = pos_ccd.split(':')
                                modifications.append({'position': int(pos), 'ccd': ccd})
                        else:
                            modifications = None
                        
                        if entity_type == 'ligand':
                            curr_dict = {entity_type: {'id'    : id,
                                                       seq_key : seq,}
                                         }
                        else:
                            curr_dict = {entity_type: {'id'    : id,
                                                       seq_key : seq.upper(),
                                                       'cyclic': cyclic}
                                         }
                        if modifications is not None:
                            curr_dict[entity_type]['modifications'] = modifications
                        
                        data_dict['sequences'].append(curr_dict)
                    
                    yaml_string = safe_dump(data_dict, sort_keys=False, indent=4)
                    yaml_string = '#This is a demo file with the ligand replaced with benzene.\n' + yaml_string
                    return yaml_string
                
                vhts_process_file_demo_button.click(write_yaml_func,
                                                    inputs=[vhts_ligand_chain_text, vhts_pocket_text,
                                                            vhts_pocket_max_distance, vhts_pocket_force_checkbox,
                                                            vhts_ligand_chain_text,
                                                            vhts_contact_1_dropdown, vhts_contact_1_text,
                                                            vhts_contact_2_dropdown, vhts_contact_2_text,
                                                            vhts_contact_max_distance, vhts_contact_force_checkbox, 
                                                            vhts_template_name_path_dict,
                                                            vhts_template_name_usage_dict,
                                                            vhts_template_name_setting_dict,
                                                            *component_refs],
                                                    outputs=vhts_yaml_demo_text)
        
        with gr.Accordion('3. Process Settings & Start Screening', open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    vhts_complex_prefix = gr.Text(label='Prefix',
                                                  info=('A prefix that will be added to the output directory '
                                                        '(quote not included)'),
                                                  placeholder='"Protein"_', interactive=True)
                    vhts_process_file_demo_button = gr.Button('Write Demo YAML')
                    vhts_start_predict_button = gr.Button('Run vHTS', interactive=False)
                vhts_yaml_demo_text = gr.Code(label='Demo YAML output',
                                              language='yaml',
                                              scale=4, interactive=False, max_lines=15)
        
        vhts_template_file.upload(read_tempaltes,
                             inputs=[vhts_template_file,
                                     vhts_template_name_chain_dict, vhts_template_name_path_dict,
                                     vhts_template_name_usage_dict, vhts_template_name_setting_dict],
                             outputs=[vhts_template_dropdown,
                                      vhts_template_name_chain_dict, vhts_template_name_path_dict,
                                      vhts_template_name_usage_dict, vhts_use_template_chekcbox,
                                      vhts_template_name_setting_dict, vhts_force_template_checkbox, vhts_template_threshold])
        vhts_use_template_chekcbox.input(update_template_chain_ids_and_settings,
                                    inputs=[vhts_use_template_chekcbox, vhts_target_chain_ids,
                                            vhts_template_chain_ids,
                                            vhts_force_template_checkbox, vhts_template_threshold,
                                            vhts_template_dropdown, vhts_template_name_usage_dict,
                                            vhts_template_name_setting_dict],
                                    outputs=[vhts_template_name_usage_dict, vhts_template_name_setting_dict])
        vhts_target_chain_ids.input(update_template_chain_ids_and_settings,
                               inputs=[vhts_use_template_chekcbox, vhts_target_chain_ids,
                                       vhts_template_chain_ids,
                                       vhts_force_template_checkbox, vhts_template_threshold,
                                       vhts_template_dropdown, vhts_template_name_usage_dict, vhts_template_name_setting_dict],
                               outputs=[vhts_template_name_usage_dict, vhts_template_name_setting_dict])
        vhts_template_chain_ids.input(update_template_chain_ids_and_settings,
                                 inputs=[vhts_use_template_chekcbox, vhts_target_chain_ids,
                                         vhts_template_chain_ids,
                                         vhts_force_template_checkbox, vhts_template_threshold,
                                         vhts_template_dropdown, vhts_template_name_usage_dict, vhts_template_name_setting_dict],
                                 outputs=[vhts_template_name_usage_dict, vhts_template_name_setting_dict])
        vhts_force_template_checkbox.input(update_template_chain_ids_and_settings,
                                 inputs=[vhts_use_template_chekcbox, vhts_target_chain_ids,
                                         vhts_template_chain_ids,
                                         vhts_force_template_checkbox, vhts_template_threshold,
                                         vhts_template_dropdown, vhts_template_name_usage_dict, vhts_template_name_setting_dict],
                                 outputs=[vhts_template_name_usage_dict, vhts_template_name_setting_dict])
        vhts_template_threshold.input(update_template_chain_ids_and_settings,
                                 inputs=[vhts_use_template_chekcbox, vhts_target_chain_ids,
                                         vhts_template_chain_ids,
                                         vhts_force_template_checkbox, vhts_template_threshold,
                                         vhts_template_dropdown, vhts_template_name_usage_dict, vhts_template_name_setting_dict],
                                 outputs=[vhts_template_name_usage_dict, vhts_template_name_setting_dict])
        vhts_template_dropdown.change(update_template_dropdown,
                                 inputs=[vhts_template_dropdown, vhts_template_name_chain_dict,
                                         vhts_template_name_usage_dict, vhts_template_name_setting_dict],
                                 outputs=[vhts_use_template_chekcbox, vhts_target_chain_ids, vhts_template_chain_ids, 
                                          vhts_force_template_checkbox, vhts_template_threshold,])
        
        vhts_boltz_log = gr.Textbox(label='Prediction Log', lines=10, max_lines=10,
                                    autofocus=False, elem_classes='log', show_copy_button=True)
        
        ligand_dataframe.change(check_yaml_strings,
                                inputs=[vhts_yaml_demo_text, vhts_complex_prefix, ligand_dataframe],
                                outputs=vhts_start_predict_button)
        vhts_complex_prefix.input(check_yaml_strings,
                                  inputs=[vhts_yaml_demo_text, vhts_complex_prefix, ligand_dataframe],
                                  outputs=vhts_start_predict_button)
        vhts_yaml_demo_text.change(check_yaml_strings,
                                   inputs=[vhts_yaml_demo_text, vhts_complex_prefix, ligand_dataframe],
                                   outputs=vhts_start_predict_button)
        vhts_start_predict_button.click(execute_vhts_boltz,
                                        inputs=[vhts_complex_prefix, ligand_dataframe, vhts_ligand_chain_text,
                                                vhts_yaml_demo_text, *all_boltz_parameters],
                                        outputs=[vhts_start_predict_button, vhts_boltz_log])
    
    
    with gr.Tab('📊 Result Visualization'):
        name_rank_f_map_state = gr.State({})
        with gr.Row():
            refresh_vis_button = gr.Button('Refresh', scale=1)
            read_vhts_checkbox = gr.Checkbox(False, label='Read vHTS Result', interactive=True)
            with gr.Column(scale=3):
                ...
        gr.Markdown('<span style="font-size:15px; font-weight:bold;">Select Name and Rank</span>')
        with gr.Row():
            result_name_dropdown = gr.Dropdown(label='Name',
                                               info='Name of the complex in the output',
                                               interactive=True)
            result_rank_dropdown = gr.Dropdown(label='Rank',
                                               info='Rank of the selected complex',
                                               interactive=True)
            result_coloring_dropdown = gr.Dropdown(label='Theme',
                                                   info='Coloring theme for structure',
                                                   choices=['chain-id', 'plddt-confidence', 'element-symbol',
                                                            'entity-id', 'entity-source', 'residue-name',
                                                            'secondary-structure', 'uniform', 'polymer-id',
                                                            'polymer-index', 'secondary-structure', 'sequence-id',
                                                            'structure-index', 'atom-id', 'molecule-type',
                                                            'hydrophobicity', 'cartoon'],
                                                   value='element-symbol',
                                                   interactive=True)
        gr.Markdown('<span style="font-size:15px; font-weight:bold;">Result</span>')
        mol_star_html = gr.HTML(get_general_molstar_html('', 0))
        with gr.Row():
            conf_df = gr.DataFrame(headers=['Metric', 'Score'], label='Overall Metrics', scale=1)
            with gr.Column(scale=2):
                with gr.Row():
                    chain_metrics = gr.DataFrame(headers=['Chain Num.', 'pTM Score'],
                                                 label='Chain pTM', scale=1)
                    pair_chain_metrics = gr.DataFrame(headers=None,
                                                      label='Pairwise chain ipTM',
                                                      show_row_numbers=True, scale=2,
                                                      wrap=True)
                aff_df = gr.DataFrame(headers=['Metric', 'Score'], label='Affinity Metrics')
        with gr.Row():
            pae_plot = gr.Plot(label='PAE', format='png')
            pde_plot = gr.Plot(label='PDE', format='png')
        plddt_plot = gr.Plot(label='pLDDT', format='png')
        
        
        refresh_vis_button.click(update_output_name_dropdown,
                                 inputs=read_vhts_checkbox,
                                 outputs=[result_name_dropdown,
                                          result_rank_dropdown,
                                          name_rank_f_map_state])
        result_name_dropdown.input(update_name_rank_dropdown,
                                   inputs=[result_name_dropdown, name_rank_f_map_state],
                                   outputs=result_rank_dropdown)
        result_name_dropdown.input(update_result_visualization,
                                   inputs=[result_name_dropdown, result_rank_dropdown,
                                           name_rank_f_map_state, result_coloring_dropdown],
                                   outputs=[mol_star_html, conf_df, chain_metrics, pair_chain_metrics,
                                            aff_df, pae_plot, pde_plot, plddt_plot])
        result_rank_dropdown.input(update_result_visualization,
                                   inputs=[result_name_dropdown, result_rank_dropdown,
                                           name_rank_f_map_state, result_coloring_dropdown],
                                   outputs=[mol_star_html, conf_df, chain_metrics, pair_chain_metrics,
                                            aff_df, pae_plot, pde_plot, plddt_plot])
        # result_coloring_dropdown.change(None, [result_coloring_dropdown], [],
        #                                 js="""
        #                                 (theme) => {
        #                                     const frame = document.getElementById('molstar_frame_general');
        #                                     frame.contentWindow.postMessage(
        #                                         { type: 'change-theme', themeName: theme },
        #                                         window.location.origin
        #                                         );}""")
        result_coloring_dropdown.input(update_general_molstar_only,
                                       inputs=[result_name_dropdown, result_rank_dropdown,
                                           name_rank_f_map_state, result_coloring_dropdown],
                                       outputs=mol_star_html)
    
    
    with gr.Tab('📈 vHTS Analysis'):
        vhts_name_df_map, vhts_name_file_map = gr.State({}), gr.State({})
        with gr.Row():
            refresh_vhts_button = gr.Button('Refresh', scale=1)
            with gr.Column(scale=3):
                ...
        
        vhts_output_options = gr.Dropdown(label='vHTS Output', multiselect=True, interactive=True)
        vhts_output_df = gr.DataFrame(label='vHTS Result Table', interactive=False,
                                      headers=['Name', 'ligand ipTM',
                                               'binding prob.',
                                               'binding aff.', 'Pass Posebusters?',
                                               'Parent'],
                                      show_row_numbers=True, show_copy_button=True, show_search='filter')
        # with gr.Row():
        #     vhts_table_download_format = gr.Dropdown(['', 'CSV', 'TSV', 'XLSX'], value='', label='Tabular Format')
        #     vhts_download_button = gr.DownloadButton('Download Tabular File', interactive=False)
        #     with gr.Column(scale=3):
        #         ...
        vhts_header = gr.Markdown('<span style="font-size:15px; font-weight:bold;">Visualization</span>')
        vhts_mol_star_html = gr.HTML(get_vhts_molstar_html('', 0))
        
        with gr.Row():
            vhts_conf_df = gr.DataFrame(headers=['Metric', 'Score'], label='Overall Metrics', scale=1)
            with gr.Column(scale=2):
                with gr.Row():
                    vhts_chain_metrics = gr.DataFrame(headers=['Chain Num.', 'pTM Score'],
                                                      label='Chain pTM', scale=1)
                    vhts_pair_chain_metrics = gr.DataFrame(headers=None,
                                                           label='Pairwise chain ipTM',
                                                           show_row_numbers=True, scale=2,
                                                           wrap=True)
                vhts_aff_df = gr.DataFrame(headers=['Metric', 'Score'], label='Affinity Metrics')
        vhts_posebust_df = gr.DataFrame(headers=['Rank', 'mol_pred_loaded', 'sanitization', 'all_atoms_connected',
                                                 'bond_lengths', 'bond_angles', 'internal_steric_clash',
                                                 'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness',
                                                 'double_bond_flatness', 'internal_energy', 'passes_valence_checks',
                                                 'passes_kekulization', 'All Passes'],
                                        label='PoseBusters Analysis', elem_classes='small-header-table')
        with gr.Row():
            vhts_pae_plot = gr.Plot(label='PAE', format='png')
            vhts_pde_plot = gr.Plot(label='PDE', format='png')
        vhts_plddt_plot = gr.Plot(label='pLDDT', format='png')
        
        
        refresh_vhts_button.click(read_vhts_directory,
                                  outputs=[vhts_name_df_map,
                                           vhts_name_file_map,
                                           vhts_output_options])
        # vhts_table_download_format.input(download_vhts_dataframe,
        #                                  inputs=[vhts_output_df, vhts_table_download_format],
        #                                  outputs=vhts_download_button)
        vhts_output_options.input(update_vhts_df_with_selects,
                                  inputs=[vhts_output_options, vhts_name_df_map],
                                  outputs=vhts_output_df)
        
        vhts_output_df.select(update_vhts_result_visualization,
                              inputs=[vhts_name_file_map],
                              outputs=[vhts_mol_star_html, vhts_conf_df, vhts_chain_metrics,
                                       vhts_pair_chain_metrics, vhts_aff_df, vhts_posebust_df,
                                       vhts_pae_plot, vhts_pde_plot, vhts_plddt_plot, vhts_header])
    
    
    with gr.Tab('📥 Boltz Output'):
        all_zipped_files_map = gr.State({})
        with gr.Row():
            refresh_button = gr.Button('Refresh', scale=1)
            with gr.Column(scale=3):
                ...
        
        with gr.Accordion('File List', open=False):
            output_file_tree = gr.FileExplorer(root_dir=output_dir,
                                               label='Output Files',
                                               interactive=True)
            
            with gr.Row():
                with gr.Column(scale=1):
                    download_selected_button = gr.Button('Download')
                    zipping_progress = gr.Text(label='Zipping Progress', interactive=False)
                download_zip_files = gr.File(label='Zipped File Download',
                                             scale=3, file_count='single',
                                             file_types=['.zip'], interactive=False)
        
        with gr.Accordion('Directory List', open=True):
            output_map = _extract_pred_dirs()
            download_file_pth_map = gr.State(output_map)
            download_ckbox_group = gr.CheckboxGroup(choices=['Structure', 'Confidence', 'Affinity', 'PAE', 'PDE', 'pLDDT', 'MSA'],
                                                    value=['Structure', 'Confidence', 'Affinity', 'PAE', 'PDE', 'pLDDT'],
                                                    label='Files to be downloaded', interactive=True)
            output_directory_options = gr.Dropdown(choices=list(output_map),
                                                   label='Output Directories',
                                                   multiselect=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    download_selected_option_button = gr.Button('Download')
                    zipping_option_progress = gr.Text(label='Zipping Progress', interactive=False)
                download_zip_option_files = gr.File(label='Zipped File Download',
                                                    scale=3, file_count='single',
                                                    file_types=['.zip'], interactive=False)
        
        refresh_button.click(update_file_tree_and_dropdown,
                             outputs=[output_file_tree,
                                      output_directory_options,
                                      download_file_pth_map])
        
        download_selected_button.click(zip_selected_files,
                                       inputs=[output_file_tree, all_zipped_files_map],
                                       outputs=[zipping_progress, download_zip_files, all_zipped_files_map])
        
        download_selected_option_button.click(zip_selected_option_files,
                                              inputs=[output_directory_options,
                                                      download_file_pth_map,
                                                      all_zipped_files_map,
                                                      download_ckbox_group],
                                              outputs=[zipping_option_progress,
                                                       download_zip_option_files,
                                                       all_zipped_files_map])
        download_zip_option_files.change(remove_zip_file,
                                         inputs=[download_zip_option_files, all_zipped_files_map],
                                         outputs=[all_zipped_files_map])
    
    
    with gr.Tab('⚙️ Boltz Paramters'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('<span style="font-size:20px; font-weight:bold;">System setting</span>')
                device_number.render()
                accelerator_type.render()
                download_model_weight = gr.Button('Download Weight (Boltz-2)')
            with gr.Column():
                gr.Markdown('<span style="font-size:20px; font-weight:bold;">Boltz Parameters</span>')
                boltz_method.render()
                recycling_steps.render()
                sampling_steps.render()
                diffusion_samples.render()
                step_scale.render()
                num_workers.render()
                preprocessing_threads.render()
                affinity_mw_correction.render()
                sampling_steps_affinity.render()
                diffusion_samples_affinity.render()
                no_kernels.render()
                override.render()
                use_potentials.render()
        download_model_weight.click(manual_download_boltz_weights, outputs=download_model_weight)
    
    
    with gr.Tab('🧰 Utilities'):
        with gr.Accordion('Inverse Complement Nucleic Acid', open=False):
            inp_nucleic_acid = gr.TextArea(label='Input DNA/RNA', lines=3, interactive=True)
            with gr.Row(equal_height=True):
                rev_comp_na_type = gr.Dropdown(['Match Input', 'DNA', 'RNA'], value='Match Input',
                                               interactive=True, label='Nucleic Acid Type', scale=1)
                rev_comp_na_text = gr.TextArea(label='Inverse Complement', lines=3,
                                               show_copy_button=True, scale=5)
            
            inp_nucleic_acid.change(reverse_complementary_nucleic_acid,
                                    inputs=[inp_nucleic_acid, rev_comp_na_type],
                                    outputs=rev_comp_na_text)
            rev_comp_na_type.change(reverse_complementary_nucleic_acid,
                                    inputs=[inp_nucleic_acid, rev_comp_na_type],
                                    outputs=rev_comp_na_text)
        
        with gr.Accordion('Display Tabular File', open=False):
            with gr.Row():
                utility_tabular_file = gr.File(label='Tabular file', interactive=True,
                                               file_types=['.csv', '.tsv', '.txt'])
                with gr.Column():
                    utility_custom_delimiter_dropdown = gr.Dropdown([r',', r'\t', r';', r' '], value=r',',
                                                                    label='Delimiter',
                                                                    allow_custom_value=True,
                                                                    interactive=True)
                    utility_read_custom_delimiter = gr.Button('Read Tabular File')
            utility_tabular_df = gr.DataFrame(label='Tabular Dataframe', interactive=False,
                                              show_row_numbers=True, show_search='filter')
            
            utility_read_custom_delimiter.click(lambda x, y: pd.read_csv(x, sep=y),
                                                inputs=[utility_tabular_file,
                                                        utility_custom_delimiter_dropdown],
                                                outputs=utility_tabular_df)
        
        with gr.Accordion('Display CCD Ligand', open=False):
            ccd_3d_ligand = gr.Text(label='CCD ID', interactive=True, info='Press Enter to submit')
            with gr.Row():
                ccd_3d_viewer = gr.HTML(get_ligand_molstar_html(''))
                ccd_3d_info = gr.DataFrame(pd.DataFrame, headers=['Type', 'Program', 'Descriptor'])
            ccd_3d_ligand.submit(draw_ccd_mol_3d, inputs=ccd_3d_ligand,
                                 outputs=[ccd_3d_viewer, ccd_3d_info])
            
        with gr.Accordion('Display SMILES Ligand', open=False):
            smiles_3d_ligand = gr.Text(label='SMILES', interactive=True, info='Press Enter to submit')
            with gr.Row():
                smiles_3d_viewer = gr.HTML(get_ligand_molstar_html(''))
                smiles_3d_info = gr.DataFrame(pd.DataFrame, headers=['Property', 'Value'],
                                              column_widths=['80%', '20%'])
            smiles_3d_ligand.submit(draw_smiles_3d, inputs=smiles_3d_ligand,
                                    outputs=[smiles_3d_viewer, smiles_3d_info])
    
    
    #####–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#####
    def change_sequence_label(curr_entity: str, sequence: str, cyclic_ckbox: bool):
        cyclic_ckbox_bool = False if curr_entity in ['CCD', 'Ligand'] else True
        return (gr.update(label=entity_label_map[curr_entity]),
                validate_sequence(curr_entity, sequence),
                gr.update(interactive=cyclic_ckbox_bool,
                          value=False if not cyclic_ckbox_bool else cyclic_ckbox),)
    
    def validate_sequence(entity_type: str, sequence: str):
        sequence = sequence.strip()
        if not sequence:
            return [('Input required!', "X")]
        if entity_type in ["Protein", "DNA", "RNA"]:
            sequence = sequence.upper()
            labeled_sequence = []
            prev_valid, prev_invalid = False, False
            allowed_chars = allow_char_dict[entity_type]
            for char in sequence:
                if char not in allowed_chars:
                    if not prev_invalid:
                        labeled_sequence.append([char, "X"])
                        prev_valid = False
                        prev_invalid = True
                    else:
                        labeled_sequence[-1][0] += char
                else:
                    if not prev_valid:
                        labeled_sequence.append([char, "✓"])
                        prev_valid = True
                        prev_invalid = False
                    else:
                        labeled_sequence[-1][0] += char
            if len(labeled_sequence) == 1 and prev_valid:
                labeled_sequence = [('Valid', "✓")]
                    
        elif entity_type == "Ligand":
            mol = Chem.MolFromSmiles(sequence)
            if mol is None:
                labeled_sequence = [(sequence, "X")]
            else:
                # labeled_sequence = [(sequence, "✓")]
                labeled_sequence = [('Valid', "✓")]
        
        elif entity_type == 'CCD':
            sequence = sequence.upper().strip()
            if not re.fullmatch(r'(?:[A-Z0-9]{3}|[A-Z0-9]{5})|[A-Z]{2}', sequence):
                labeled_sequence = [(sequence, "X")]
            else:
                # labeled_sequence = [(sequence, "✓")]
                labeled_sequence = [('Valid', "✓")]
                
        return labeled_sequence
    
    def update_chain_seq_dict(entity_type: str, chain: str, seq: str, old_dict: dict):
        if not all((chain, seq)):
            return old_dict, gr.update(), gr.update()
        old_dict.update({chain: {'type'    : entity_type,
                                 'sequence': seq,}})
        return old_dict, gr.update(choices=list(old_dict)), gr.update(choices=list(old_dict))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Boltz Gradio interface")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing (share=True)")
    parser.add_argument("--inbrowser", action="store_true", help="Start Gradio in current default browser (inbrowser=True)")
    args = parser.parse_args()
    
    threading.Thread(target=concurrent_download_model_weight, daemon=True).start()
    Interface.launch(server_name="0.0.0.0", server_port=7860, share=args.share, inbrowser=args.inbrowser)