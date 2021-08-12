import numpy as np

def create_scheduler(args):
    if args.sched == "step":
        def lr_scheduler(epoch,lr):
            if (epoch+1) % args.step_size == 0:
                lr = lr*args.gamma

            return lr
        return lr_scheduler
    elif args.sched == "cosine":
        def lr_scheduler(epoch,lr):
            return args.eta_min + 0.5*(args.lr - args.eta_min)*(1 + np.cos(epoch * np.pi / args.T_max))

        return lr_scheduler
    else:
        return None
