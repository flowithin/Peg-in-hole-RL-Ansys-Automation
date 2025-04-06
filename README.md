- This is a project that utilizes reinforcement learning to solve peg-in-hole problem in a typical setting.
- Due to rise of complexity, it will be heavy workload to formulate equations in different scenarios.
- To fascilitate RL, we use ANSYS to simulate the peg-in-hole process with different material and shapes.
- Notably, we formalize a workflow in python automation inside the ANSYS software feeding data into RL training models.
How to obtain reaction force?

$$
F_{sp}=2\pi R^2\times9\times stress=443N
$$

APDL: 

[ANSYS命令流里常用命令 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/431207876)

Extract everage stress: 

```python
/post1
set,last

ETABLE,str_elem,S,EQV                               ! Stores stress information for elements
ETABLE,vol_elem,VOLU,                               ! Stores the volume of each element
SMULT,weighted_elemstress,str_elem,vol_elem,1,1,    ! This step multiplies each element stress by corresponding volume
SSUM                                                ! Simply sum up the ETABLE entities
*get,total_weighted_elemstress,ssum,,item,weighted_elemstress   ! Stores sum of weighted element stress
*get,total_vol,ssum,,item,vol_elem                              ! Stores sum of total volume
my_volume_average_elementstress = total_weighted_elemstress/total_vol   ! Calculates volume weighted average of element stress
fini
```

Extract nodal forces:

```APDL
! Define a node set for the surface of interest (modify this accordingly)
NSLV,1,SURF,,1

! Extract normal forces using /POST26 command
/POST1
ETABLE,NFORC,,SUM
NFORC,ALL,SURF,,SUM
NSOL,1
PLNSOL,U,SURF
```

[Using Python in ANSYS Mechanical](https://blog.ozeninc.com/resources/using-python-in-ansys-mechanical-to-search-the-tree-generate-scripts)

pyMAPDL

```python
import pymapdl as ansys

# Start a connection to an already running ANSYS instance
mapdl = ansys.launch_mapdl()

# Load your results file (make sure to specify the correct file name)
result_file = 'path/to/your/resultfile.rst'
mapdl.finish()  # Exit the interactive mode
mapdl.resume
mapdl.cdworkingdirectory('/path/')

# Specify the output file (CSV format)
output_file = 'output.csv'

# Define the items you want to export (nodal displacements in this case)
items = 'U'

# Retrieve the nodal solution data
nodal_solution = mapdl.post_processing(result_file)

# Export nodal solution data to a CSV file
nodal_solution.to_csv(output_file, items=items, node='N')

# Print a confirmation message
print(f"Nodal solution data has been exported to {output_file}")

# Exit ANSYS
mapdl.exit()
```

Generate txt

```python
#Search for and list all Construction Geometry Objects

for i, item in enumerate(Model.Children):

   if item.GetType()==Ansys.ACT.Automation.Mechanical.ConstructionGeometry:

       for j, cg in enumerate(item.Children):

           print str(j) + ", " + cg.Name


#Solve the existing analysis

Model.Analyses[0].Solve()

#Get the Solution Object

sol=Model.Analyses[0].Solution



#Search for and output text results for all LinearizedEquivalentStress Objects

for i, item in enumerate(sol.Children):

   if item.GetType()==Ansys.ACT.Automation.Mechanical.Results.LinearizedStressResults.LinearizedEquivalentStress:

       location=item.Location

        filename="D:\Work_Ozen\LinearizedStress_" + item.Name + "_" + location.Name + ".txt"

       item.ExportToTextFile(filename)

       print "Generated File: " + chr(34) + filename + chr(34)
```

Write to csv file

```python
import csv



def writeCSV(filename, data):

   # Function to write python list to a csv file

   with open(filename, 'wb') as csvfile:

       spamwriter = csv.writer(csvfile, delimiter=',',

                               quotechar='|', quoting=csv.QUOTE_MINIMAL)

       for row in data:

           spamwriter.writerow(row)



ResultsOfInterest = []

ResultsOfInterest.append('Linearized Equivalent Stress')





import wbjn

cmd = 'returnValue(GetUserFilesDirectory())'

user_dir = wbjn.ExecuteCommand(ExtAPI, cmd)



AnalysisNumber=0



solution=Model.Analyses[AnalysisNumber].Solution

for j, item in enumerate(solution.Children):

   if item.GetType() == Ansys.ACT.Automation.Mechanical.Results.LinearizedStressResults.LinearizedEquivalentStress:

       if item.Name in ResultsOfInterest:

         item.Activate()



           data=[]

           del data[:]

           Pane=ExtAPI.UserInterface.GetPane(MechanicalPanelEnum.TabularData)

           Con = Pane.ControlUnknown

           for R in range(1,Con.RowsCount+1):

              data.append([])

               for C in range(2,Con.ColumnsCount+1):

                   data[-1].append(Con.cell(R,C).Text)



           writeCSV(user_dir + "/" + Model.Analyses[AnalysisNumber].Name + " - " + item.Name + ".csv", data)



print("Script has completed!")

print("")

print("Open File: " + chr(34) + user_dir + chr(92) + Model.Analyses[AnalysisNumber].Name + " - " + item.Name + ".csv" + chr(34))
```

A simple method

```python
Model.Analyses[0].Solve()
sol = Model.Analyses[0].Solution
for i, item in enumerate(sol.Children):
    if (item.Name[0] == 'N'): 
        print(sol.Children[i].Average)
```

Applicable way to export data:

- Method [0] : 
1. Use <u>ExportToTextfile</u> funtion to export **nodal stress** to a txt file.

2. Copy the data to a exel file

3. use python to calculate average value

Why are the stress results negative? 

> Typically, positive normal stress is tensilestress and negative normal stress is *compressive*

datatype: quantity

apply force

- ASEL ，LOC, X.../NSEL

- Python scripting select by ID

- PyMAPDL script

linear fitting model

- SVM

- linear

- NN

$(x,y,z,\theta_x,\theta_y,\theta_z)$

$(F_{xpos},F_{ypos},F_{zpos},F_{xmag},F_{ymag},F_{zmag})$
