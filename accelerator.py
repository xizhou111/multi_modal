from accelerate import Accelerator

class MyAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logging_dir(self):
        return '/mnt/cfs/NLP/zcl/multi_modal/logs'