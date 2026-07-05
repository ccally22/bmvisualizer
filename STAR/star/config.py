import os

# Config file for STAR body model
# Defines paths to the STAR model files (male, female, neutral)
# Paths are relative so they work on all operating systems (Windows, Mac, Linux)

class cfg:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Path to male STAR model
    path_male_star    = os.path.join(base, 'data', 'body_models', 'star', 'star_male.npz')
    # Path to female STAR model
    path_female_star  = os.path.join(base, 'data', 'body_models', 'star', 'star_female.npz')
    # Path to neutral STAR model
    path_neutral_star = os.path.join(base, 'data', 'body_models', 'star', 'star_neutral.npz')

# Create instance so "from config import cfg" works
cfg = cfg()
