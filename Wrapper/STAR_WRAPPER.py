from Wrapper.body_model_wrapper import BodyModelWrapper
from STAR.star.pytorch.star import STAR

class STAR_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender):
        self._model = STAR(gender=gender.lower())
        return self

    def forward(self, input_params, betas):
        # berechnung der neuen werte
        for k, v in input_params.items():
            input_params[k] = v.reshape(1, -1)

        model_output = self._model(
            betas=betas,
            #expression=self._body_exp_tensor,
            **input_params,
        )
        verts = model_output.vertices[0].detach().numpy()
        joints = model_output.joints[0].detach().numpy()
        faces = self._model.faces

        return verts, joints, faces