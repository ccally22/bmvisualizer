import torch
import numpy as np
from MHR.mhr.mhr import MHR
from Wrapper.body_model_wrapper import BodyModelWrapper

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
NUM_MODEL_PARAMETERS = 204
LOD = 1

class MHR_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender=None):
        self._model = MHR.from_files(device=torch.device("cpu"), lod=LOD)
        self._identity_coeffs = torch.zeros(1, NUM_IDENTITY_BLENDSHAPES)
        self._face_expr_coeffs = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES)
        self._faces = np.asarray(self._model.character.mesh.faces, dtype=np.int32)
        return self

    def forward(self, input_params, betas=None):
        if betas is not None:
            self._identity_coeffs = betas.reshape(1, NUM_IDENTITY_BLENDSHAPES)
        # pose
        model_parameters = input_params['model_parameters']

        with torch.no_grad():
            verts, skel_state = self._model(
                self._identity_coeffs,
                model_parameters,
                self._face_expr_coeffs,
            )

        verts = verts.squeeze(0).detach().numpy()
        joints = skel_state.squeeze(0)[:, :3].detach().numpy()
        faces = self._faces

        mesh_data = (verts, joints, faces)
        return mesh_data