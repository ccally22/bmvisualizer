import torch
import numpy as np
from MHR.mhr.mhr import MHR
from Wrapper.body_model_wrapper import BodyModelWrapper

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
NUM_MODEL_PARAMETERS = 204
LOD = 1
# idk was das ist
UNBOUNDED_SENTINEL = 1.0e30
UNBOUNDED_FALLBACK = 3.14

class MHR_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender=None):
        self._model = MHR.from_files(device=torch.device("cpu"), lod=LOD)
        self._identity_coeffs = torch.zeros(1, NUM_IDENTITY_BLENDSHAPES)
        self._face_expr_coeffs = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES)
        self._faces = np.asarray(self._model.character.mesh.faces, dtype=np.int32)

        pt = self._model.character.parameter_transform
        self._pose_param_names = list(pt.names[:NUM_MODEL_PARAMETERS])

        mins, maxs = self._model.character.model_parameter_limits
        mins = np.asarray(mins[:NUM_MODEL_PARAMETERS], dtype=np.float32)
        maxs = np.asarray(maxs[:NUM_MODEL_PARAMETERS], dtype=np.float32)
        mins = np.where(mins < -UNBOUNDED_SENTINEL, -UNBOUNDED_FALLBACK, mins)
        maxs = np.where(maxs > UNBOUNDED_SENTINEL, UNBOUNDED_FALLBACK, maxs)
        self._pose_param_limits = np.stack([mins, maxs], axis=1)

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

        verts = verts.squeeze(0).detach().numpy() / 100.0
        joints = skel_state.squeeze(0)[:, :3].detach().numpy() / 100.0
        faces = self._faces

        return (verts, joints, faces)