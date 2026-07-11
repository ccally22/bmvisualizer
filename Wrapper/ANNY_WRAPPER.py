import os
import sys
import roma
import torch
import numpy as np

ANNY_SRC_DIR = os.path.join(os.path.dirname(__file__), "anny", "src")
if os.path.isdir(ANNY_SRC_DIR) and ANNY_SRC_DIR not in sys.path:
    sys.path.insert(0, ANNY_SRC_DIR)

from anny.models.full_model import create_model

from Wrapper.body_model_wrapper import BodyModelWrapper


class ANNY_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender):
        self._model = create_model()
        return self

    def forward(self, input_params, shape_params=None):
        # shape_params bei ANNY: dict mit phenotypes
        phenotypes = shape_params if shape_params else {}
        # ohne den ersten 1 teil (der immer gleich ist):
        rotvec = input_params['pose'][0]
        # erstellt die rotationsmatrix
        bones_rotmat = roma.rotvec_to_rotmat(rotvec)
        # rotation + translation = rotation aus den input parametern + 0 + der erste teil wird wieder hinzugefügt
        # Translation aus input_params holen (nur Root-Bone bekommt sie)
        trans_array = torch.zeros((len(bones_rotmat), 3), dtype=torch.float64)
        if 'trans' in input_params and input_params['trans'].numel() > 0:
            trans_array[0] = input_params['trans'][0, 0].to(torch.float64)

        pose_parameters = roma.Rigid(
            bones_rotmat, trans_array
        )[None].to_homogeneous()

        model_output = self._model(
            pose_parameters=pose_parameters,
            phenotype_kwargs=phenotypes,
            local_changes_kwargs={},
            pose_parameterization=None,
            return_bone_ends=False
        )
        # anny gibt dictionary zurück keine Objekte, deshalb muss man anders darauf zugreifen
        verts = model_output["vertices"].squeeze(0).detach().numpy()
        # die joints werden bei anny unter bone_poses gespeichert
        joints = (
                model_output["bone_poses"][0, :, :3, 3]
                .detach()
                .numpy()
        )
        # torch tensor der noch konvertiert werden muss zu numpy array
        faces = self._model.get_triangular_faces().cpu().numpy().astype(np.int32)

        R = roma.euler_to_rotmat('x', [270.], degrees=True).numpy()
        verts = verts @ R.T
        joints = joints @ R.T

        mesh_data = (verts, joints, faces)

        return mesh_data
