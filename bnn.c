/*


*/
#include<stdlib.h>
#include<stdio.h>
#include<string.h>

void show_help(char *, char *);
int bnn_init(char *);
int bnn_train(char *);


#define OFF 0
#define ON 1
#define PAUSED 2
#define RESET 3
#define ADD 4
#define REMOVE 5
#define SHOW 6

#define BNN_MAX_NODES 10
#define BNN_MAX_CONNECTIONS 3

#define DEFAULT_INPUTS 2
#define DEFAULT_OUTPUTS 1
#define DEFAULT_NODES 2
#define DEFAULT_LAYERS 3	// input(1) + hidden(n) + output(n+1)

struct _node {
	int value;
	int bias;
	int inputs;
	int weights;
	int signoid;
};

int main(int argc, char *argv[]) {

	int counter;
	int status = OFF;
	int node_count = DEFAULT_INPUTS+DEFAULT_NODES*(DEFAULT_NODES-2)+DEFAULT_OUTPUTS;
	int ret_val = 0;

	char *filename = NULL;
	

	struct _node node[node_count];

	//if (argc < 4) {
	//	printf("Error\n");
	//	exit(1);
	//}

	filename = argv[1]; // malloc strlen(argv[1])+4 and add extenion '.cfg'
	//if (strcmp(argv[2], "help") == 0) {
	if (argc < 4) {
		show_help(argv[1], argv[2]);
		return 1;
	}	
	else if (strcmp(argv[2], "init") == 0) {
		int inputs = DEFAULT_INPUTS;
		int outputs = DEFAULT_OUTPUTS;
		int nodes = DEFAULT_NODES; 
		int layers = DEFAULT_LAYERS;
		ret_val = bnn_init(argv[3]);
		return ret_val;
	}
	else if (strcmp(argv[2], "train") == 0) {
		ret_val = bnn_train(argv[3]);
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

	for (counter=1; counter < argc; counter++) {
		if (!strcmp(argv[counter],"start")) status = ON;
		
	}


	printf("node:%d %d\n",node[0].value,node[0].bias);
	return 0;
}

int bnn_init(char *source) {
	printf("init: %s\n", source);
	return 0;
}

int bnn_train(char *source) {
	printf("train: %s\n", source);
	return 0;
}

void show_help(char *arg1, char *arg2) {

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
			printf("         signoid = function to calculate node value (Default: 0)\n\n");
			printf("        name of a file containing network definitions in format\n");
			printf("         [i|number|o],bias,inputs,weights,signoid\n\n");
			printf("         inputs and weights are binary encoded, so '1,5,15,3,0\\n1,-3,15,8,0'\n");
			printf("         sets node 1 and 2 in hidden layer 1 connected to four input nodes\n");
			printf("         (1111=15) and uses weights -1,-1,-1,1 (0001=8, 0=-1, 1=1)\n\n");
			printf("        Type 'bnn help signoid' for a list of available functions.\n\n");
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
			printf("        Input values can be supplied via a file or on the command line.\n\n");
			printf("        File format: inputs,outputs,nodes,layers,signoid\n\n");
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
			printf("        0 = 1/x where x=\n\n");
		}
	}

	return; 
}
