
def create_scheduler(args):
    if args.sched == "step":
        def lr_scheduler(epoch,lr):
            if (epoch+1) % args.step_size == 0:
                lr = lr*args.gamma

            return lr
        return lr_scheduler

    else:
        return None
