# from omegaconf import OmegaConf

from pipeline import Pipeline


def main():
    # Config = OmegaConf.load('config.yaml')
    
    Pipeline('config.yaml',
             remove_prev_runs=True,
             prepare=False,
             train=True,
             export=False,).run()


if __name__ == '__main__':
    main()
