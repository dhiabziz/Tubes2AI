# BACA INI DULU BUAT DEVELOPER
## SEBELUM NGE PUSH, PASTIIN CLEAR ALL OUTPUTS DULU (BIAR RAPI, GA GAMPANG KONFLIK, JADI JANGAN PERNAH NGE PUSH OUTPUT)
## BIKIN BRANCH DENGAN NAMA BRANCH IALAH FITUR YANG DIKERJAKAN (EX: preprocess, )

# Tubes2AI
Tugas Besar 2 AI - Implementasi Algoritma Pembelajaran Mesin

---

## **Folder Structure**

```
. 
└── Tubes2AI/
    ├── .venv/          # Local virtual environment (hidden by default)
    ├── src/            # Source code folder
    ├── README.md       # Project documentation
    └── requirements.txt # List of project dependencies
```

## **Setup Instructions**

### 1. **Create a Virtual Environment**
To isolate project dependencies, create a virtual environment in the Tubes2AI folder:
- Open a terminal and navigate to the project directory:
  ```bash
  cd Tubes2AI
  ```
- Create a virtual environment locally named `.venv`:
  ```bash
  python -m venv .venv
  ```
  This will create a `.venv` folder in the project directory containing the isolated Python environment.

### 2. **Activate the Virtual Environment**
Once the virtual environment is created, activate it.
- **On Windows**:
  ```bash
  .\.venv\Scripts\activate
  ```
  When activated, the terminal prompt will change to show the environment name, e.g., `(venv)`.

### 3. **Install Dependencies**
After activating the environment, install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. **Add a Dependency**
To add a new Python package to your environment:
1. Install the package:
   ```bash
   pip install <package_name>
   ```
   Example:
   ```bash
   pip install numpy
   ```

2. Update the `requirements.txt` file to include the new dependency (Necessary if you added new dependency):
   ```bash
   pip freeze > requirements.txt
   ```

### 5. **Deactivate the Virtual Environment**
When done working in the virtual environment, deactivate it with:
```bash
deactivate
```
This will return you to the global Python environment.

### 6. **Tips for Managing Dependencies**
- **View Installed Packages**:
  ```bash
  pip list
  ```
- **Manually Edit (Not recommended) `requirements.txt`**: You can open and edit `requirements.txt` to add/remove specific packages.
- **Avoid Committing `.venv` to Version Control**: Add `.venv` to `.gitignore` to prevent it from being uploaded to Git repositories.

## **Notes**
- Always activate the `.venv` before running your Python scripts to ensure the correct dependencies are used.
- Use `requirements.txt` to maintain consistency across environments.