import csv
'''
def AddPressure(x,y,MAG):
    #adding pressure at a specified named selection
    pressure = Model.Analyses[0].AddPressure() # Add a pressure.
    #selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
    loc = str(x)+','+str(y)
    ASEL = ExtAPI.DataModel.GetObjectsByName(loc)
    pressure.Location = ASEL[0]
    Magnitude = str(MAG) +' '+'[MPa]'
    #pressure.Magnitude.Inputs[0].DiscreteValues = [Quantity("0 [s]"), Quantity("1 [s]")]  # Set the time values.
    #pressure.Magnitude.Output.DiscreteValues = [Quantity("10 [Pa]"), Quantity("20 [Pa]")]  # Set the magnitudes.

    pressure.Magnitude.Inputs[0].DiscreteValues = [Quantity("0 [s]"), Quantity("1 [s]")]
    pressure.Magnitude.Output.DiscreteValues = [Quantity("0 [Pa]"), Quantity(Magnitude)]'''

# Location storing
loc = []
# NodeID storing, each location correspond to the node Id at that point
nodeId = []

Inputfile = 'D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\output.csv'
def Add_force_ByID(Id, Xcom, Ycom, Zcom, force):
        '''Given an Id apply force'''
        node = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
        node.Ids = [Id]
        force.Name = 'F at' + str(Id)
        force.DefineBy = LoadDefineBy.Components
        force.Location = node
        force.XComponent.Output.SetDiscreteValue(0, Quantity(Xcom, "N"))
        force.YComponent.Output.SetDiscreteValue(0, Quantity(Ycom, "N"))
        force.ZComponent.Output.SetDiscreteValue(0, Quantity(Zcom, "N"))
def writeCSV(filename, data):
# Function to write python list to a csv file
    with open(filename, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator = '\n')
        spamwriter.writerow(data)
# Read nodeID from the a given file
def ReadNodeId(filename, nodeId, loc):
    '''read a group of node Ids'''
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0][0] == ':':
                loc.append(row[0])
            else: 
                nodeId.append(row[0])
def TryOut(index):
    '''Argument: Id-the Id of the mesh node\\
       This function is a integrated method to
       1. Create a force 
       2. Add a force with function "Add_force_ByID"
       3. Solve
       4. Output to file with "F at Id"'''
    Id = int(nodeId[index])
    force = Model.Analyses[0].AddForce()
    Add_force_ByID(Id,5,6,7, force)
    # Solve
    Model.Analyses[0].Solve()
    ResultsOfInterest = []
    for i in range(2, 55):   
        name = 'Normal Stress ' + str(i)
        ResultsOfInterest.append(name)
    AnalysisNumber=0
    solution = Model.Analyses[AnalysisNumber].Solution
    for j, item in enumerate(solution.Children):
    # if item.GetType() == Ansys.ACT.Automation.Mechanical.Results.LinearizedStressResults.LinearizedNormalStress:
        if item.Name in ResultsOfInterest:
        # print('yes')
            item.Activate()
            data=[]
            del data[:]
            Pane=ExtAPI.UserInterface.GetPane(MechanicalPanelEnum.TabularData)
            Con = Pane.ControlUnknown
            for R in range(1,Con.RowsCount+1):
                data.append([])
                for C in range(2,Con.ColumnsCount+1):
                    data[-1].append(Con.cell(R,C).Text)
       # print(data)
            writeCSV('D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\Pythontest\\'+'F at '+ loc[index][1:] +'.csv', data[-1])
    force.Delete()
    print('D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\Pythontest'+'F at '+ loc[index][1:] +'.csv successfully created')
ReadNodeId(Inputfile, nodeId, loc)
print(nodeId)
print(loc)
for i in range(0,2):
    TryOut(i)
print("Script has completed!")
print("")

