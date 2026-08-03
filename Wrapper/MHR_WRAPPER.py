import torch
import numpy as np
from MHR.mhr.mhr import MHR
from Wrapper.body_model_wrapper import BodyModelWrapper

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
NUM_MODEL_PARAMETERS = 204
LOD = 1
# idk was das isti
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
        self._joint_names = list(self._model.character.skeleton.joint_names)

        mins, maxs = self._model.character.model_parameter_limits
        mins = np.asarray(mins[:NUM_MODEL_PARAMETERS], dtype=np.float32)
        maxs = np.asarray(maxs[:NUM_MODEL_PARAMETERS], dtype=np.float32)
        mins = np.where(mins < -UNBOUNDED_SENTINEL, -UNBOUNDED_FALLBACK, mins)
        maxs = np.where(maxs > UNBOUNDED_SENTINEL, UNBOUNDED_FALLBACK, maxs)
        self._pose_param_limits = np.stack([mins, maxs], axis=1)

        return self

    def forward(self, input_params, shape_params=None, expression=None):
        if shape_params is not None:
            n = min(shape_params.shape[1], NUM_IDENTITY_BLENDSHAPES)
            self._identity_coeffs[0, :n] = shape_params[0, :n]

        # Face Expression falls übergeben
        if expression is not None:
        #if 'expression' in input_params:
        #    exp = input_params['expression']
            n = min(expression.shape[1], NUM_FACE_EXPRESSION_BLENDSHAPES)
            self._face_expr_coeffs[0, :n] = expression[0, :n]
            
        # Model parameters kopieren
        model_parameters = input_params['model_parameters'].clone()

        # Root Translation (Index 0, 1, 2)
        if 'trans' in input_params and input_params['trans'].numel() > 0:
            trans = input_params['trans'][0, 0]
            model_parameters[0, 0] = trans[0]
            model_parameters[0, 1] = trans[1]
            model_parameters[0, 2] = trans[2]

        # Root Rotation (Index 3, 4, 5)
        if 'global_orient' in input_params:
            rot = input_params['global_orient'][0, 0]
            model_parameters[0, 3] = rot[0]
            model_parameters[0, 4] = rot[1]
            model_parameters[0, 5] = rot[2]

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