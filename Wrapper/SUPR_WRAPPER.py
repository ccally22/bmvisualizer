from Wrapper.body_model_wrapper import BodyModelWrapper
import 

class SUPR_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender):
        self._model = SUPR(f'data/body_models/supr/supr_{gender}.npy')
        return self

    def forward(self, input_params, shape_params, expression=None):
        # berechnung der neuen werte
        for k, v in input_params.items():
            input_params[k] = v.reshape(1, -1)

        model_output = self._model(
            betas=shape_params,
            #expression=self._body_exp_tensor,
            **input_params,
        )
        verts = model_output.vertices[0].detach().numpy()
        joints = model_output.joints[0].detach().numpy()
        faces = self._model.faces

        return verts, joints, faces
