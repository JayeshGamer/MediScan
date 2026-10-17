import time
import torch
import numpy as np
from evaluate_tumornet import load_trained_model as load_tumor, GradCAM as TumorGradCAM
from evaluate_pneumonet import load_trained_model as load_pneumo, GradCAM as PneumoGradCAM
from evaluate_gastronet import load_trained_model as load_gastro, GradCAM as GastroGradCAM

torch.set_num_threads(4) # Standard realistic 4-thread deployment setting on CPU
device = torch.device('cpu')

models = {
    'TumorNet': load_tumor('checkpoints/TumorNet_best.pth', device)[0],
    'PneumoNet': load_pneumo('checkpoints/PneumoNet_best.pth', device)[0],
    'GastroNet': load_gastro('checkpoints/GastroNet_best.pth', device)[0]
}

gradcams = {
    'TumorNet': TumorGradCAM(models['TumorNet'], device),
    'PneumoNet': PneumoGradCAM(models['PneumoNet'], device),
    'GastroNet': GastroGradCAM(models['GastroNet'], device)
}

results = {}

for mname, model in models.items():
    model.eval()
    gc = gradcams[mname]
    
    print(f'Benchmarking {mname}...', flush=True)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            dummy = torch.randn(1, 3, 224, 224, device=device)
            _ = model(dummy)
    
    # Mode 1: Single Pass (no TTA, no Grad-CAM)
    times_single = []
    with torch.no_grad():
        for _ in range(25):
            x = torch.randn(1, 3, 224, 224, device=device)
            t0 = time.perf_counter()
            _ = model(x)
            times_single.append((time.perf_counter() - t0) * 1000.0)
    print(f'  Single pass mean: {np.mean(times_single):.1f} ms', flush=True)

    # Mode 2: 4-Pass TTA (no Grad-CAM)
    times_tta = []
    with torch.no_grad():
        for _ in range(25):
            x = torch.randn(1, 3, 224, 224, device=device)
            t0 = time.perf_counter()
            _ = model(x)
            _ = model(torch.flip(x, dims=[3]))
            h, w = 224, 224
            x_crop = torch.nn.functional.interpolate(x[:, :, 6:h-6, 6:w-6], size=(h, w), mode='bilinear', align_corners=False)
            _ = model(x_crop)
            _ = model(torch.flip(x_crop, dims=[3]))
            times_tta.append((time.perf_counter() - t0) * 1000.0)
    print(f'  4-Pass TTA mean: {np.mean(times_tta):.1f} ms', flush=True)

    # Mode 3: Single Pass + Grad-CAM
    times_cam = []
    for _ in range(25):
        x = torch.randn(1, 3, 224, 224, device=device)
        t0 = time.perf_counter()
        _ = gc.generate(x.clone())
        times_cam.append((time.perf_counter() - t0) * 1000.0)
    print(f'  Single + Grad-CAM mean: {np.mean(times_cam):.1f} ms', flush=True)

    # Mode 4: 4-Pass TTA + Grad-CAM
    times_full = []
    for _ in range(25):
        x = torch.randn(1, 3, 224, 224, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            o1 = model(x)
            o2 = model(torch.flip(x, dims=[3]))
            h, w = 224, 224
            x_crop = torch.nn.functional.interpolate(x[:, :, 6:h-6, 6:w-6], size=(h, w), mode='bilinear', align_corners=False)
            o3 = model(x_crop)
            o4 = model(torch.flip(x_crop, dims=[3]))
        _ = gc.generate(x.clone())
        times_full.append((time.perf_counter() - t0) * 1000.0)
    print(f'  4-Pass TTA + Grad-CAM mean: {np.mean(times_full):.1f} ms', flush=True)

    results[mname] = {
        'single': (float(np.mean(times_single)), float(np.percentile(times_single, 95))),
        'tta': (float(np.mean(times_tta)), float(np.percentile(times_tta, 95))),
        'cam': (float(np.mean(times_cam)), float(np.percentile(times_cam, 95))),
        'full': (float(np.mean(times_full)), float(np.percentile(times_full, 95)))
    }

print('\n' + '=' * 70, flush=True)
print('  MEDISCAN AI LATENCY BENCHMARK (CPU: Intel i7-13700HX, 4 threads, BS=1)', flush=True)
print('=' * 70, flush=True)
for mname, res in results.items():
    print(f'{mname}:', flush=True)
    print(f'  Single forward pass (no TTA, no CAM) : {res["single"][0]:.1f} ms (p95: {res["single"][1]:.1f} ms)', flush=True)
    print(f'  4-pass TTA (no CAM)                   : {res["tta"][0]:.1f} ms (p95: {res["tta"][1]:.1f} ms)', flush=True)
    print(f'  Single forward + Grad-CAM             : {res["cam"][0]:.1f} ms (p95: {res["cam"][1]:.1f} ms)', flush=True)
    print(f'  4-pass TTA + Grad-CAM (Full Pipeline) : {res["full"][0]:.1f} ms (p95: {res["full"][1]:.1f} ms)', flush=True)
