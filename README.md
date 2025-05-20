# Peg-in-Hole Reinforcement Learning with ANSYS Simulation

This project uses reinforcement learning (RL) to solve the classic **peg-in-hole** insertion problem in robotic or mechanical assembly. Given the increasing complexity of analytical formulations under varying materials and geometries, we adopt a simulation-based approach to train RL models effectively.

---

## Overview

- Leverages **ANSYS simulations** to model peg-in-hole interactions with different shapes and materials.
- Automates simulation workflows using **Python scripting** within ANSYS Mechanical.
- Extracts physical parameters like **reaction force** and **stress** to feed into RL models.
- Facilitates RL training with realistic simulation data for improved generalization.

---

## Why Simulation?

Analytically modeling contact mechanics in the peg-in-hole setting is increasingly difficult for complex shapes, friction, and material deformation. This project:

- Uses ANSYS for reliable contact mechanics simulation.
- Automates stress and force extraction using **pyMAPDL** and ANSYS scripting.
- Feeds results into an RL agent for training insertion policies.

---

## Key Equation

To estimate the reaction force:

$$
F_{sp} = 2\pi R^2 \times 9 \times \text{stress} = 443 \text{ N}
$$

---

## ANSYS Commands for Force & Stress Extraction

### Volume-Averaged Stress (APDL)

```apdl
/post1
set,last

ETABLE,str_elem,S,EQV
ETABLE,vol_elem,VOLU
SMULT,weighted_elemstress,str_elem,vol_elem,1,1
SSUM
*get,total_weighted_elemstress,ssum,,item,weighted_elemstress
*get,total_vol,ssum,,item,vol_elem

my_volume_average_elementstress = total_weighted_elemstress / total_vol

fini
```

### Nodal Force Extraction (APDL)

```apdl
NSLV,1,SURF,,1

/POST1
ETABLE,NFORC,,SUM
NFORC,ALL,SURF,,SUM
NSOL,1
PLNSOL,U,SURF
```

Reference: [ANSYS Common Commands](https://zhuanlan.zhihu.com/p/431207876)

---

## ANSYS Python Scripting

### Extract Nodal Displacement using pyMAPDL

```python
import pymapdl as ansys

mapdl = ansys.launch_mapdl()
mapdl.finish()
mapdl.resume
mapdl.cdworkingdirectory('/path/')

result_file = 'path/to/your/resultfile.rst'
output_file = 'output.csv'
items = 'U'

nodal_solution = mapdl.post_processing(result_file)
nodal_solution.to_csv(output_file, items=items, node='N')

print(f"Nodal solution data has been exported to {output_file}")
mapdl.exit()
```

### Export Text Results using ACT Scripts

```python
# Get linearized equivalent stress and export to .txt
for i, item in enumerate(Model.Analyses[0].Solution.Children):
    if item.GetType() == Ansys.ACT.Automation.Mechanical.Results.LinearizedStressResults.LinearizedEquivalentStress:
        location = item.Location
        filename = f"D:/Work_Ozen/LinearizedStress_{item.Name}_{location.Name}.txt"
        item.ExportToTextFile(filename)
        print(f"Generated File: \"{filename}\"")
```

### Export Tabular Data to CSV

```python
import csv

def writeCSV(filename, data):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

ResultsOfInterest = ['Linearized Equivalent Stress']

import wbjn
cmd = 'returnValue(GetUserFilesDirectory())'
user_dir = wbjn.ExecuteCommand(ExtAPI, cmd)

solution = Model.Analyses[0].Solution

for item in solution.Children:
    if item.GetType() == Ansys.ACT.Automation.Mechanical.Results.LinearizedStressResults.LinearizedEquivalentStress:
        if item.Name in ResultsOfInterest:
            item.Activate()
            data = []
            pane = ExtAPI.UserInterface.GetPane(MechanicalPanelEnum.TabularData)
            control = pane.ControlUnknown
            for R in range(1, control.RowsCount + 1):
                row_data = [control.cell(R, C).Text for C in range(2, control.ColumnsCount + 1)]
                data.append(row_data)

            filename = f"{user_dir}/{Model.Analyses[0].Name} - {item.Name}.csv"
            writeCSV(filename, data)
            print(f"Exported to {filename}")
```

---

## Simple Stress Reader

```python
Model.Analyses[0].Solve()
sol = Model.Analyses[0].Solution
for i, item in enumerate(sol.Children):
    if item.Name.startswith('N'): 
        print(item.Average)
```

---

## Data Extraction Workflows

### Method 0: Manual Export and Postprocessing

1. Use `ExportToTextFile` to export **nodal stress** as `.txt`.
2. Copy data into Excel.
3. Use Python or spreadsheet to compute average stress values.

---

## Notes on Stress Sign

- Positive stress: **Tensile**
- Negative stress: **Compressive**

---

## Simulation Control Techniques

- Node/element selection: `ASEL`, `NSEL`, or by Python ID selection
- Data export via pyMAPDL or ACT scripts

---

## Learning Model Targets

Input variables:

- Pose/State:  
  $(x, y, z, \theta_x, \theta_y, \theta_z)$

Output targets:

- Force feedbacks:  
  $(F_{xpos}, F_{ypos}, F_{zpos}, F_{xmag}, F_{ymag}, F_{zmag})$

---

## ML Model Candidates

- **SVM** (Support Vector Machines)
- **Linear regression**
- **Neural Networks**

