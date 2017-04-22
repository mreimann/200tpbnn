/*
* BNN - BINARY NETWORK SIMULATOR
*
* The Binary Network Simulator allows the setup, training and execution of a neural network 
* with 3 to 16 layers and 1 to 16 nodes per layer.*
*  
* This program was written as part of the SJSU CmpE200 semester project by Team .
*
* Team Members: 
*
* Note: Compile with gcc -o bnn bnn.c -lm
*
* ToDo:	Add functionality to visualize network [bnn name show] (?)
*	Expand SET for non-full networks (?)
*	Enfore Upper dimension for values (32) (?)
*	Check if filename has extension and, if not, append .cfg extension 
*/

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>

int show_help(char *, char *);
int bnn_init(char *, int, char **, int, int *, int);
int bnn_train(char *);
int bnn_run(char *);

#define FALSE 0
#define TRUE 1

#define BNN_INIT 0
#define BNN_TRAIN 1
#define BNN_RUN 2

#define DIM_INPUTS 0
#define DIM_OUTPUTS 1
#define DIM_NODES 2
#define DIM_LAYERS 3 
#define DIM_SIGNOID 4

#define DEFAULT_INPUTS 2	// Input Nodes
#define DEFAULT_OUTPUTS 1	// Output Nodes
#define DEFAULT_NODES 2		// Nodes per hidden layer
#define DEFAULT_LAYERS 1	// Number of hidden layers
#define DEFAULT_SIGNOID 1
#define DEFAULT_BIAS 1
#define DEFAULT_WEIGHT 1

#define DELIM ","

struct _node {
	int value;
	int bias;
	int inputs;
	int weights;
	int signoid;
};

int main(int argc, char *argv[]) {

	int counter;
	int node_count = DEFAULT_INPUTS+DEFAULT_NODES*(DEFAULT_NODES-2)+DEFAULT_OUTPUTS;
	int ret_val = 0;
	int argpos = 1;
	int mode;
	int isVerbose = FALSE;

	int dimensions[5] = {DEFAULT_INPUTS, DEFAULT_OUTPUTS, DEFAULT_NODES, DEFAULT_LAYERS, DEFAULT_SIGNOID};

	char *filename = NULL;
	

	struct _node node[node_count];

	if (argc < 4) {
		ret_val = show_help(argv[1], argv[2]);
		return ret_val;
	}
	else {
		filename = argv[argpos];		
		argpos++;

		if (strcmp(argv[argpos], "init") == 0) {
			mode = BNN_INIT;
		}
		else if (strcmp(argv[argpos], "train") == 0) {
			mode = BNN_TRAIN;
		}
		else if (strcmp(argv[argpos], "run") == 0) {
			mode = BNN_RUN;
		}
		else {
			ret_val = show_help(argv[1], argv[2]);
			return ret_val;
		}
		argpos++;

		if (strcmp(argv[argpos], "verbose") == 0) {
			isVerbose = TRUE;
			argpos++;
		}
	}

	if (BNN_INIT == mode) {
		ret_val = bnn_init(filename, argc, argv, argpos, dimensions, isVerbose);		
		return ret_val;
	}
	else if (BNN_TRAIN == mode) {
		ret_val = bnn_train(filename);
		return ret_val;
	}
	else {
		ret_val = bnn_run(filename);
		return ret_val;
	}
	


	// init nodes
	for (counter=0; counter < node_count; counter++) {
		node[counter].value = 0;
		node[counter].bias = 10;
		node[counter].inputs = 0;
		node[counter].weights = 0;
		node[counter].signoid=0;
	}

	printf("node:%d %d\n",node[0].value,node[0].bias);
	return 0;
}

int bnn_init(char *target, int argc, char *argv[], int argpos, int *dim, int isVerbose) {

	FILE *fp;
	char *source = argv[argpos];
	int isFull = (strcmp(source, "full") == 0);
	int counter = 0;
	int total_nodes = 0;	
	int non_outpout_nodes = 0;	
	char type;
	int number;
	int max_nodes = 4 * (int)sizeof(int);

	argpos++; 
	
	if (isFull) {
		for (counter = argpos; counter < argc; counter++) {
			dim[counter - argpos] = atoi(argv[counter]);
			if (0 == dim[counter - argpos]) {
				printf("Bad input value %s.\n\n", argv[counter]);
				return EXIT_FAILURE;
			}
		}
	}
	else if (strcmp(source, "set") == 0) {
		char *strings[5] = {"input nodes", "output nodes", "nodes per hidden layer", "hidden layers", "signoid function"};
		char input[3];

		printf("\nEnter values for network size:\n\n");

		printf(" Network type (0=Full, 1=Partial): 0\n\n");
		printf(" Note: At this time, 'set' only supports full networks. To define a partial\n");
		printf("       network, edit the inputs in file '%s' manually.\n\n", target);
		isFull = TRUE;

		do {
			printf(" Number of %s (1-%d): ", strings[counter], max_nodes);
			scanf ("%s", input);
			dim[counter] = atoi(input);
			if (dim[counter] > 0 && dim[counter] <= max_nodes) counter++; 
			
		} while (counter < 5);		
	}
	else {
		fp = fopen(source, "r");
		char buffer[255];
		char *token;

		if (!fp) {
			printf("Invalid filename: %s\n\n", source); 
			return EXIT_FAILURE;
		}

		fscanf(fp, "%s", buffer);
		token = strtok(buffer, DELIM);
		isFull = (strcmp(token, "full") == 0);

		// Did the file define network as full?
		if (isFull) {
			while (token && counter < 4) {
				token = strtok(NULL, DELIM);
				dim[counter] = atoi(token);
				
				if (0 == dim[counter]) {
					printf("Bad input value %s in %s.\n\n",token, source);
					return EXIT_FAILURE;	
				}

				counter++;
			}
			fclose(fp);
		}
		else {
			printf("Bad input file %s (input file only supports full).", source);
			return EXIT_FAILURE;	
		}
   	}	

	if (isVerbose) {	
		printf("Initializing %snetwork '%s' with properties:\n", (isFull?"full ":""), target);
		printf(" Inputs: %d\n", dim[DIM_INPUTS]);
		printf(" Outputs: %d\n", dim[DIM_OUTPUTS]);
		printf(" Nodes: %d\n", dim[DIM_NODES]);
		printf(" layers: %d\n", dim[DIM_LAYERS]);
		printf(" Signoid: %d\n\n", dim[DIM_SIGNOID]);
	}

	fp = fopen(target, "w+");
	if (!fp) {
		printf("Unable to write definitions file: %s\n\n", target); 
		return EXIT_FAILURE;
	}

	for (counter = 0; counter < dim[DIM_INPUTS]; counter++) {
		fprintf(fp, "i,0,0,%d,0\n", DEFAULT_WEIGHT);
	}
	
	for (counter = 0; counter < (dim[DIM_LAYERS] * dim[DIM_NODES]); counter++) {
		number = (int)pow(2,(counter<dim[DIM_NODES]?dim[DIM_INPUTS]:dim[DIM_NODES])) - 1;
		fprintf(fp, "%d,%d,%d,%d,%d\n",(int)(counter/dim[DIM_NODES]),DEFAULT_BIAS,number,(DEFAULT_WEIGHT?number:0),dim[DIM_SIGNOID]);
	}

	for (counter = 0; counter < dim[DIM_OUTPUTS]; counter++) {
		number = (int)pow(2,dim[DIM_NODES]) - 1;
		fprintf(fp, "o,%d,%d,%d,%d\n",DEFAULT_BIAS,number,(DEFAULT_WEIGHT?number:0),dim[DIM_SIGNOID]);
	}
	fclose(fp);

	return 0;
}

int bnn_train(char *source) {
	printf("train: %s\n", source);
	return 0;
}

int bnn_run(char *source) {
	printf("run: %s\n", source);
	return 0;
}

int show_help(char *arg1, char *arg2) {

	int isHelp = ( arg1 && strcmp(arg1, "help") == 0 );

	if (isHelp) printf("\nBinary Network Simulator\n\n");

	printf("Usage: bnn help|name action [verbose] args\n\n");

	if (isHelp) {
		if (!arg2) {
			printf("       name    (required) = name of network\n");
			printf("       action  (required) = operation. Valid options = {init, train, run}\n");
			printf("       verbose (optional) = show calculations\n\n");
			printf("       source             = file with source values\n");
			printf("       set                = prompt for values\n");
			printf("       values             = numberic values provided on command line\n\n");
			printf("       Type 'bnn help argument_name' for detailed help\n\n");
		}
		else if ( strcmp(arg2, "name") == 0 ) {
			printf("       'name' is a required argument.\n\n");
			printf("       The name denotes the instance of a neural network to load.\n");
			printf("       Netork definitions are saved as name.cfg.\n\n");
		}
		else if ( strcmp(arg2, "action") == 0 ) {
			printf("       'action' is a required argument.\n\n");
			printf("       Valid choices for actions are:\n\n");
			printf("        init  = define the structure of a new neural network.\n");
			printf("        train = train a defined neural network.\n");
			printf("        run   = use the network to evaluate a set of input values.\n\n");
			printf("       Type 'bnn help action_name' for details.\n\n");
		}
		else if ( strcmp(arg2, "verbose") == 0 ) {
			printf("       'verbose' is an optional argument.\n\n");
			printf("       When verbose is set, all calculations are echoed to the terminal.\n\n");
		}
		else if ( strcmp(arg2, "args") == 0 ) {
			printf("        'args' is a required argument\n\n");
			printf("        Valid choices for args are:\n\n");
			printf("         full   = Create a fully linked neural network.\n");
			printf("                  Note: Only available for action 'init'.\n");
			printf("         source = a source file with input values.\n");
			printf("         set    = prompt for input values.\n");
			printf("         values = Provide input values on the command line.\n");
			printf("                  Note: Only available for actions 'init' and  'run'.\n\n");
			printf("         Type 'bnn help args_name' for details.\n\n");
		}
		else if ( strcmp(arg2, "init") == 0 ) {
			printf("       action 'init': Initialize a neural network. Valid args are:\n\n");
			printf("        full [inputs] [output] [nodes] [layers] [signoid] where\n");
			printf("         full   = flag to create a fully connected network\n");
			printf("         inputs = number of input nodes (Default: 2)\n");
			printf("         outputs = number of output nodes (Default: 1)\n");
			printf("         nodes   = number of nodes in each hidden layer (Default: 2)\n");
			printf("         layers  = number of layers including input and output (Default: 3)\n");
			printf("         signoid = function to calculate node value (Default: 0)\n");
			printf("         (Type 'bnn help signoid' for a list of available functions.)\n");
			printf("         (Type 'bnn help full' for 'full' file format.)\n\n");	
			printf("        name of a file containing network definitions in format\n");
			printf("         [i|number|o],bias,inputs,weights,signoid\n\n");
			printf("         inputs and weights are binary encoded, so '1,5,15,3,0\\n1,-3,15,8,0'\n");
			printf("         sets node 1 and 2 in hidden layer 1 connected to four input nodes\n");
			printf("         (1111=15) and uses weights -1,-1,-1,1 (0001=8, 0=-1, 1=1)\n\n");
		}
		else if ( strcmp(arg2, "train") == 0 ) {
			printf("        action 'train': Set up the network using a file of known inputs and\n");
			printf("                        outputs.\n\n");
			printf("        File format: i(1,1),...,i(1,max_input):o(1,1),...,o(1,max_output\n");
			printf("                     (...)\n");
			printf("                     i(n,1),...,i(n,max_output):o(n,1),...,o(n,max_output)\n\n");
		}
		else if ( strcmp(arg2, "run") == 0 ) {
			printf("        action 'run': Evaluate a set of input values using a trained network.\n\n");
			printf("        Input values can be supplied via a file or on the command line.\n\n");
		
		}
		else if ( strcmp(arg2, "full") == 0 ) {
			printf("        arg 'full': Initialize a fully connected neural network. In a\n");
			printf("                    fully connected neural network, each layer n (n>1) node\n");
			printf("                    is connected to each layer n-1 node.\n\n");
			printf("        Input values can be supplied via a file or on the command line. If a\n");
			printf("        file is used, supply the filename instead of 'full' and use the file\n");
			printf("        format: full,inputs,outputs,nodes,layers,signoid\n\n");
			printf("        Node: If some values are omitted, defaults are used.\n\n");
			printf("        For command line option or default values, type 'bnn help init'.\n\n");
		}
		else if ( strcmp(arg2, "source") == 0 ) {
			printf("       arg 'source': Use a file to provide input values. File format differ\n");
			printf("                     based on the action type. In some cases, a file can be\n");
			printf("                     used in conjunction with values. In this case, values\n");
			printf("                     overwrite the parameters provided in the file.\n\n");
			printf("       For file formats, type 'bnn help action'.\n\n");
		}
		else if ( strcmp(arg2, "set") == 0 ) {
			printf("       arg 'set': Prompt for input values during action\n\n");
		}
		else if ( strcmp(arg2, "values") == 0 ) {
			printf("       arg 'values': Provide input values on the command line. In some cases,\n");
			printf("                     values can be used in conjunction with a file. In this\n");
			printf("                     case, values overwrite the parameters provided in the file.\n\n");
			printf("       For arguments, type 'bnn help action'.\n\n");
		}
		else if ( strcmp(arg2, "signoid") == 0 ) {
			printf("       A signoid is a function used in calculating the value of a neural network\n");
			printf("       node. Different signoid functions can be selected. Valid choices are:\n\n");
			printf("        1 = 1/x where x=\n\n");
		}
	}

	return isHelp?EXIT_SUCCESS:EXIT_FAILURE; 
}
