$ErrorActionPreference = 'Continue'
Write-Host "PHASE 2"
Write-Host "COMMAND: python -m compileall src scripts tests"
python -m compileall src scripts tests
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import src'"
python -c 'import src'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'from src.training.train_pipeline import train'"
python -c 'from src.training.train_pipeline import train'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'from src.training.data_module import SkinLesionDataModule'"
python -c 'from src.training.data_module import SkinLesionDataModule'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'from src.modules.classification.classifier import SkinLesionClassifier'"
python -c 'from src.modules.classification.classifier import SkinLesionClassifier'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "PHASE 3"
Write-Host "COMMAND: pip check"
pip check
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python --version"
python --version
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import torch; print(torch.__version__)'"
python -c 'import torch; print(torch.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import torchvision; print(torchvision.__version__)'"
python -c 'import torchvision; print(torchvision.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import timm; print(timm.__version__)'"
python -c 'import timm; print(timm.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import albumentations; print(albumentations.__version__)'"
python -c 'import albumentations; print(albumentations.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)'"
python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import numpy; print(numpy.__version__)'"
python -c 'import numpy; print(numpy.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"

Write-Host "COMMAND: python -c 'import sklearn; print(sklearn.__version__)'"
python -c 'import sklearn; print(sklearn.__version__)'
Write-Host "EXIT CODE: $LASTEXITCODE"
