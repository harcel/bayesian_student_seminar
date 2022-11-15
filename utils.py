import pandas as pd

def make_ex_line(posterior):
    """ Make a finely sampled array along x that follows the maximum a posteriori.
        posterior should be trace.posterior from a sampled PyMC model. This is specifically meant
        to deal with the "constant+linear model" situation in the notebook.
        
        Returns x and y arrays to plot.
    """
    
    x_post = np.linspace(0,5, 1000)
    
    # Those x_post - x_post thing os to make the dimensions work. Ugly but it works...
    constpart = x_post - x_post + posterior["Constant"].mean().values
    linpart = x_post * posterior["Slope"].mean().values + \
            posterior["Switchpoint"].mean().values*posterior["Slope"].mean().values - \
            posterior["Slope"].mean().values
    
    y_post = pm.math.switch(at.gt(posterior["Switchpoint"].mean().values, x_post), 
                        constpart, linpart)
    
    return x_post, y_post
                        

    
    
def read_student_data():
    pd.read_csv()
    