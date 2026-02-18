import mmcv
import torch


def single_gpu_test(model, data_loader, show=False, show_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        if show or show_dir:
            model.module.show_results(
                data,
                result,
                out_dir=show_dir,
                show=show,
            )

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
