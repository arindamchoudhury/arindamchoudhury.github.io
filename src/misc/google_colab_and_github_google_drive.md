## Google Colab, Github and Google Drive

- Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

- Go to your directory in Google Drive:
```python
%cd drive/MyDrive/ColabNotebooks
```

- clone your github project:
```python
!git clone https://github.com/arindamchoudhury/project.git
```

- Your project files will stay in Google drive
```python
%cd project
!git pull
```

