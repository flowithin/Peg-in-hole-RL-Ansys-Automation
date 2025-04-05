import csv
# Location storing
loc = []
# NodeID storing, each location correspond to the node Id at that point
nodeId = []

Inputfile = 'D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\output 1228 data.csv'
dataOutputfile = 'D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\Pythontest\\dataOutput_disp0.1mm1_4_9.csv'
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
def Clearfile(filename):
    with open(filename,'w') as f:
        f.truncate()
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
                loc.append(row[0][1:])
            else: 
                nodeId.append(row[0])
                #print (row[0])
def TryOut(index):
    '''Argument: Id-the Id of the mesh node\\
       This function is a integrated method to
       1. Create a force 
       2. Add a force with function "Add_force_ByID"
       3. Solve
       4. Output to file with "F at Id"'''
    locline = []
    locline.append(loc[index])
    writeCSV(dataOutputfile,locline)
    if nodeId[index] != 'null':
        force = Model.Analyses[0].AddForce()
        Id = int(nodeId[index])
        Add_force_ByID(Id,-4,0,0, force)
    # Solve
    Model.Analyses[0].Solve()
    ResultsOfInterest = []
    # Generate names of interest
    for x in ['R', 'L']:
        for i in range(1, 4):
            for j in range(1,4):   
                name = 'Force Reaction ' + x + str(i) + str(j)
                ResultsOfInterest.append(name)
            
    print(ResultsOfInterest)
    AnalysisNumber=0
    solution = Model.Analyses[AnalysisNumber].Solution
    for j, item in enumerate(solution.Children):
    # if item.GetType() == Ansys.ACT.Automation.Mechanical.Results.LinearizedStressResults.LinearizedNormalStress:
        if item.Name in ResultsOfInterest:
            #print('yes')
            item.Activate()
            data=[]
            del data[:]
            Pane=ExtAPI.UserInterface.GetPane(MechanicalPanelEnum.TabularData)
            Con = Pane.ControlUnknown
            for R in range(1, Con.RowsCount+1):
                data.append([])
                for C in range(2,Con.ColumnsCount+1):
                    data[-1].append(Con.cell(R,C).Text) 
       # print(data)
            writeCSV(dataOutputfile, data[-1])
    if nodeId[index] != 'null': 
        force.Delete()
    #print('D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\Pythontest\\'+'F at '+ loc[index] +'.csv successfully created')
ReadNodeId(Inputfile, nodeId, loc)
Clearfile(dataOutputfile)
firstline = [['Location'],['Fx'],['Fy'],['Fz'],['Ftot']]
writeCSV(dataOutputfile, firstline)
#print(nodeId)
#print(loc)
for i in range(13504,15601):
   TryOut(i)
   #print(i)
#TryOut(0)
print("Script has completed!")
print("")
