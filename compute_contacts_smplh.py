"""
Code to generate contact labels from SMPL and object registrations
Author: Xianghui Xie
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os, re
import numpy as np
sys.path.append(os.getcwd())
import trimesh
import igl
from os.path import join, isfile
from data.frame_data import FrameDataReader
from viz.contact_viz import ContactVisualizer

from data.kinect_transform import KinectTransform

# imports for rendering, you can replace with your own code
from viz.pyt3d_wrapper import Pyt3DWrapper
import pytorch3d

import pickle as pkl
import cv2

from glob import glob
import json

from tqdm import tqdm

from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from collections import defaultdict

class ContactLabelGenerator(object):
    "class to generate contact labels"
    def __init__(self):
        pass
     
   
    def get_contact_labels(self, smpl, obj, num_samples, thres=0.02):
        """
        sample point on the object surface and compute contact labels for each point
        :param smpl: trimesh object
        :param obj: trimesh object
        :param num_samples: number of samples on object surface
        :param thres: threshold to determine whether a point is in contact with the human
        :return:
        for each point: a binary label (contact or not) and the closest SMPL vertex
        """
        object_points = obj.sample(num_samples)

        # dist: Distance between each object point to closest smpl faces
        # idx: Each object point's closest face id
        dist, face_ids, _ = igl.signed_distance(object_points, smpl.vertices, smpl.faces, return_normals=False)

        # If distance is smaller than threshold, define as contact
        contact = dist < thres

        return object_points, contact, face_ids[contact]

    def to_trimesh(self, mesh):
        tri = trimesh.Trimesh(mesh.v, mesh.f, process=False)
        return tri
    
def check_directory_valid(input_directory):
    if not os.path.isdir(input_directory):
        raise ValueError(f"{input_directory} is not a valid directory")
    
    # Get the basename of the directory
    basename = os.path.basename(input_directory)

    # Define the regex pattern
    pattern = r'^Date\d{2}_Sub\d{2}_.+'

    data_pths = []
    
    if basename == 'sequences':
        seq_names = os.listdir(input_directory)

        data_pths = []
        for name in seq_names:
            if re.match(pattern, name):
                data_pths.append(os.path.join(input_directory, name))

        if len(data_pths) == 0:
            raise ValueError(f"{input_directory} is empty" )
        
        return data_pths
    # Check if the basename matches the pattern
    if not re.match(pattern, basename):
        raise ValueError(f"{input_directory} is not a valid directory" )
    
    data_pths.append(input_directory)

    return data_pths


def make_vertex_face_mapping(faces, contact_vert_indices):
    # contact_vert_indices = np.unique(faces).tolist()

    vertice_face_mapping = {}
    for vert_id in contact_vert_indices:

        # True if face contains vert id
        mask = (faces == vert_id).any(-1)
        face_indices = np.where(mask)[0]

        vertice_face_mapping[vert_id] = face_indices.tolist()


    return vertice_face_mapping

def draw_points(img, points):
    for i in range(points.shape[0]):
        x, y = points[i]
        cv2.circle(img, (int(x), int(y)), color=(0,0,255), radius=2, thickness=2)


def get_contact_parts(contact_vertices, smpl_vert_seg):
    part_names = list(smpl_vert_seg.keys())

    cnts = np.zeros(len(part_names))

    for vert_id in contact_vertices:
        for i, part_name in enumerate(part_names):
            if vert_id in smpl_vert_seg[part_name]:
                cnts[i] += 1
    
    res = [part_names[i] for i in range(len(part_names)) if cnts[i] > 0]
    return res

def compute_normalized_bounding_box(mask):
    # Find the coordinates of the non-zero elements
    non_zero_coords = np.argwhere(mask)
    
    # Get the min and max coordinates along both axes
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0)
    
    # Normalize the coordinates
    H, W = mask.shape
    x_min_norm = x_min / W
    x_max_norm = x_max / W
    y_min_norm = y_min / H
    y_max_norm = y_max / H
    
    # Return the normalized bounding box as (x_min_norm, y_min_norm, x_max_norm, y_max_norm)
    return x_min_norm, y_min_norm, x_max_norm, y_max_norm


def main(args):


    current_directory = os.path.dirname(os.path.abspath(__file__))

    smpl_vert_seg = json.load(open(os.path.join(current_directory, 'support_data/smpl_vert_segmentation.json'), 'r'))

   

    save_res = []

    seq_paths = check_directory_valid(args.seq_folder)

    have_contact_cnt = 0
    for seq_i, seq_path in enumerate(seq_paths):
        

        basename = os.path.basename(seq_path)
        print(f'Processing {seq_i+1}/{len(seq_paths)}', basename)
        reader = FrameDataReader(seq_path)
        batch_end = reader.cvt_end(args.end)
        generator = ContactLabelGenerator()
        smpl_fit_name, obj_fit_name = 'fit02', 'fit01'
        # contact_vizer = ContactVisualizer()
        # kinect_transform = KinectTransform(seq_name, kinect_count=reader.kinect_count)


        res_list = []

        for idx in tqdm(range(args.start, batch_end)):
            outfile = reader.objfit_meshfile(idx, obj_fit_name).replace('.ply', '_contact.npz')
            # if isfile(outfile) and not args.redo:
            #     print(outfile, 'done, skipped')
            #     continue
            smpl = reader.get_smplfit(idx, smpl_fit_name)
            obj = reader.get_objfit(idx, obj_fit_name)

            smpl_params = reader.get_smplfit_params(idx, smpl_fit_name)

            smpl_trimesh = generator.to_trimesh(smpl)
            obj_trimesh = generator.to_trimesh(obj)
            

            # contacts: (N, ) For each sampled object point, True if there is a contact
            # vertices: (N, 3) For each sampled object point, the nearest contacted vertice coordinate
            # faces_ids: (N, ) For each sampled object point, the nearest contacted face id
            samples, contacts, contact_faces_ids = generator.get_contact_labels(
                smpl_trimesh, obj_trimesh, args.num_samples, thres=args.thres
            )
           
            # print(smpl_trimesh.vertices.shape, smpl_trimesh.faces.shape)
            # print(smpl_trimesh.vertices)

           
            ts = reader.frames[idx]
            
            if len(contact_faces_ids) > 0: 
                contact_faces_ids = np.unique(contact_faces_ids)
                contact_faces = np.array(smpl_trimesh.faces)[contact_faces_ids]
                contact_vert_indices = np.unique(contact_faces)
                # points2d = kinect_transform.project2color(vertices[contacts], kid)

                # For each contact vertice
                vert_to_face_mapping = make_vertex_face_mapping(smpl_trimesh.faces, contact_vert_indices)

                # contact_faces = contact_faces.tolist()

                # contact_vert_indices = set(np.array(faces)[contacts].tolist())
                contact_parts = get_contact_parts(contact_vert_indices, smpl_vert_seg)

                

            else:
                contact_faces_ids = None
                contact_vert_indices = None
                contact_parts = None
                vert_to_face_mapping = None
                contact_faces = None
                

            res_list.append({
                'ts': ts,
                'contact_vert_indices': contact_vert_indices,
                'contact_faces': contact_faces,
                'contact_face_ids': contact_faces_ids,
                'contact_parts': contact_parts,
                'vert_to_face_mapping': vert_to_face_mapping,
                'smplh': {'pose': smpl_params[0], 'betas': smpl_params[1], 'trans': smpl_params[2]},
                'smpl_vert_coords': np.array(smpl_trimesh.vertices),

            })

            print(res_list[-1])

        if args.out is None:
            out_pth = seq_path
        else:
            out_pth = args.out
            os.makedirs(out_pth, exist_ok=True)

        save_pth = os.path.join(out_pth, 'contact_smplh_data.pkl')
        with open(save_pth, 'wb') as f:
            pkl.dump(res_list, f)

        print(f'Results saved to {save_pth}')

    print('all done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-fs', '--start', type=int, default=0, help='index of the start frame')
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('--thres', type=float, default=0.02, help='Distance threshold to bo considered as contact')
    parser.add_argument('-o', '--out', type=str, default=None, help='Save directory. If not specified, results will be stored in the original directories')

    args = parser.parse_args()

    main(args)





