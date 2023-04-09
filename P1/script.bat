set PYTHON_EXE=venv\Scripts\python.exe

%PYTHON_EXE% P1.py .\Instancias_APC\diabetes_1.arff .\Instancias_APC\diabetes_2.arff .\Instancias_APC\diabetes_3.arff .\Instancias_APC\diabetes_4.arff .\Instancias_APC\diabetes_5.arff > resultados-diabetes.txt


%PYTHON_EXE% P1.py .\Instancias_APC\ozone-320_1.arff .\Instancias_APC\ozone-320_2.arff .\Instancias_APC\ozone-320_3.arff .\Instancias_APC\ozone-320_4.arff .\Instancias_APC\ozone-320_5.arff > resultados-ozone.txt


%PYTHON_EXE% P1.py .\Instancias_APC\spectf-heart_1.arff .\Instancias_APC\spectf-heart_2.arff .\Instancias_APC\spectf-heart_3.arff .\Instancias_APC\spectf-heart_4.arff .\Instancias_APC\spectf-heart_5.arff > resultados-heart.txt