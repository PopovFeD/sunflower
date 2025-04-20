pyinstaller main.py --exclude-module PyQt6

pyinstaller main.py \
  --exclude-module torch.distributed \
  --exclude-module torch.utils.benchmark \
  --exclude-module torch.testing \
  --exclude-module torch.cuda \
  --exclude-module torch.backends.mps \
  --exclude-module torch.ao \
  --exclude-module torch._inductor \
  --exclude-module torch.fx \
  --exclude-module torch.onnx

pyinstaller main.py --exclude-module PyQt6 --exclude-module torch.distributed --exclude-module torch.utils.benchmark --exclude-module torch.testing --exclude-module torch.cuda --exclude-module torch.backends.mps --exclude-module torch.ao --exclude-module torch._inductor --exclude-module torch.fx --exclude-module torch.onnx