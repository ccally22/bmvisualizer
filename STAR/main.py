from star.ch.star import STAR
import numpy as np

model = STAR(gender='female',num_betas=10)
## Assign random pose and shape parameters
model.pose[:] = np.random.rand(model.pose.size) * .2
model.betas[:] = np.random.rand(model.betas.size) * .03

for j in range(0,10):
    model.betas[:] = 0.0  #Each loop all PC components are set to 0.
    for i in np.linspace(-3,3,10): #Varying the jth component +/- 3 standard deviations
        model.betas[j] = i