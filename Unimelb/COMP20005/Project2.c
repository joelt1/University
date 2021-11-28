/* Submission for COMP20005 Project 2, 2019 Semester 1. Context - identifying
   historical trends that may exist in rainfall records, and allowing
   visualisation of the data.
   
   Authorship Declaration:

   (1) I certify that the program contained in this submission is completely
   my own individual work, except where explicitly noted by comments that
   provide details otherwise.  I understand that work that has been developed
   by another student, or by me in collaboration with other students,
   or by non-students as a result of request, solicitation, or payment,
   may not be submitted for assessment in this subject.  I understand that
   submitting for assessment work developed by or in collaboration with
   other students or non-students constitutes Academic Misconduct, and
   may be penalized by mark deductions, or by other penalties determined
   via the University of Melbourne Academic Honesty Policy, as described
   at https://academicintegrity.unimelb.edu.au.

   (2) I also certify that I have not provided a copy of this work in either
   softcopy or hardcopy or any other form to any other student, and nor will
   I do so until after the marks are released. I understand that providing
   my work to other students, regardless of my intention or any undertakings
   made to me by that other student, is also Academic Misconduct.

   (3) I further understand that providing a copy of the assignment
   specification to any form of code authoring or assignment tutoring
   service, or drawing the attention of others to such services and code
   that may have been made available via such a service, may be regarded
   as Student General Misconduct (interfering with the teaching activities
   of the University and/or inciting others to commit Academic Misconduct).
   I understand that an allegation of Student General Misconduct may arise
   regardless of whether or not I personally make use of such solutions
   or sought benefit from such actions.

   Signed by: Joel Thomas 915951
   Dated:     12/05/2019
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_LINES 3000			/* Maximum 12*250 months of data for array */
#define START 0					/* Index of first data row in dataset */
#define IMPOSSIBLE_VAL -1		/* Impossible value for a year in dataset */
#define FIRST_MONTH 1			/* First month of the year (January) */
#define LAST_MONTH 12			/* Last month of the year (December) */
#define LEN_MONTHS 13			/* Array length for months, nothing at pos 0 */
#define NO_YEAR_FOUND 0			/* When required year not found in dataset */
#define NUM_CHARS_MONTH 3		/* Characters to represent months e.g. "Jan" */
#define NUMER 2					/* Numerator of formula for Kendall's T */
#define NO 0					/* Waiting till required condition met  */
#define YES 1					/* Signal to start creating bar plot */
#define MAX_HEIGHT 24			/* Maximum number of rows in bar plot */
#define LAST_ROW 0				/* Last row before numbered label v reaches 0 */
#define FIRST_ARG 1				/* Index of first year as an argument for S4 */
#define SECOND_LAST 2			/* String index for second last digit in year */
#define LAST 3					/* String index for last digit in year */

/* Months of the year as strings contained in an array */
char *months[] = {"", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
	"Sep", "Oct", "Nov", "Dec"};

/* Struct to hold station number, year, month, rainfall and validated data
   values scanning of data */
typedef struct {
	int station_num;			/* Station number of rainfall site */
	int year;					/* Year in which rainfall was recorded */
	int month;					/* Month in which rainfall was recorded */
	double rainfall;			/* Amount of rainfall in given month and year */
	char validated;				/* Indicates whether data has been validated */
} row_t;

/* Array of structs given a predefined maximum size */
typedef row_t data_t[MAX_LINES];

/* Function prototypes */
int my_getchar(void);
void discard_chars(void);
int read_rows(data_t data_array);
void S1_output(int num_rows, row_t (*p_data)[]);
int first_year(int num_rows, row_t (*p_data)[], int req_month);
int last_year(int num_rows, row_t (*p_data)[], int req_month);
double sum_calc(int num_rows, row_t (*p_data)[], int req_month);
int num_values_calc(int num_rows, row_t (*p_data)[], int req_month);
void S2_output(int num_rows, row_t (*p_data)[], double mean_vals[]);
void S3_output(int num_rows, row_t (*p_data)[]);
double max_rain_point(int num_rows, row_t (*p_data)[], int year,
	double mean_vals[]);
void init_arr(int *p_arr, int arr[], int length_arr);
double rain_given_year_month(int num_rows, row_t (*p_data)[], int year,
	int month);
void print_bar_plot(int num_rows, row_t (*p_data)[], int year,
	int print_check_arr[], int scale_factor, double mean_vals[],
	char last_two_digits[], double max);
void print_data_bar_plot(int num_rows, row_t (*p_data)[], int year, int v,
	int print_check_arr[], int scale_factor, double mean_vals[],
	char *p_two_digits);
void print_second_last_row(void);
void print_last_row(void);
void S4_output(int num_rows, row_t (*p_data)[], int argc, char*argv[],
	double mean_vals[]);


/* Combining all functions and forming backbone of program */
int
main(int argc, char *argv[]) {
	int num_rows;							/* Number of data rows in file */
	double mean_vals[LEN_MONTHS];			/* Array to store mean values of
											   rainfall for each month */
	data_t data;							/* Array of structs for dataset */
	discard_chars();
	num_rows = read_rows(data);
	row_t (*p_data)[num_rows];				/* Pointer to array of structs
											   (data)*/
	p_data = &data;
	/* Prints output for Stage 1 */
	S1_output(num_rows, p_data);
	printf("\n");
	/* Prints output for Stage 2 */
	S2_output(num_rows, p_data, mean_vals);
	printf("\n");
	/* Prints output for Stage 3 */
	S3_output(num_rows, p_data);
	printf("\n");
	/* Prints output for Stage 4 */
	S4_output(num_rows, p_data, argc, argv, mean_vals);
	printf("Ta daa!\n");
	return 0;
}

/* Equivalent of getchar() but for cross system usage */
int
my_getchar(void) {
	int i;
	while ((i=getchar())=='\r') {
	}
	return i;
}

/* Discards header (first row) in data and any remaining characters in rows of
   data to allow scanf on actual data */ 
void
discard_chars(void) {
	while (my_getchar() != '\n');
}

/* Reads each data row in dataset into an array of structs
   Parameters:
   data_array = array of structs to hold different characteristics of data */
int
read_rows(data_t data_array) {
	int row_num = START,		/* Buddy variable for array */
	station_val,				/* Station number of rainfall site */
	year_val,					/* Year in which rainfall was recorded */
	month_val;					/* Month in which rainfall was recorded */
	double rain_val;			/* Amount of rainfall in given month and year */
	char valid_val;				/* Indicates whether data has been validated */
	while (scanf("IDCJAC0001,%d,%d,%d,%lf,%c",
			&station_val, &year_val, &month_val, &rain_val, &valid_val) == 5) {
		if (row_num == MAX_LINES) {
			/* Exit program if given too much data */
			printf("Too many data lines, please try again.\n");
			exit(EXIT_FAILURE);
		}
		/* Defining individual characteristics of each struct in array */
		data_array[row_num].station_num = station_val;
		data_array[row_num].year = year_val;
		data_array[row_num].month = month_val;
		data_array[row_num].rainfall = rain_val;
		data_array[row_num].validated = valid_val;
		row_num++;
		/* Skip any remaining characters until newline character appears */
		discard_chars();
	}
	return row_num;
}

/* Prints output for Stage 1
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset */
void
S1_output(int num_rows, row_t (*p_data)[]) {
	int n = START, month, site_num, curr_year;
	/* Extract site number from dataset, doesn't change throughout dataset */
	site_num = (*p_data)[START].station_num;
	printf("S1, site number 0%d, %d datalines in input",site_num,num_rows);
	/* Set value of current year to impossible value to ensure first if
	   statement below always executes in first loop */
	curr_year =  IMPOSSIBLE_VAL;
	while (n < num_rows) {
		/* Set current year equal to year from dataset */
		if ((*p_data)[n].year != curr_year) {
			curr_year = (*p_data)[n].year;
			month = FIRST_MONTH;
			printf("\nS1, %d:   ", curr_year);
		}
		/* If data does not start from January, fill with "..." */
		while (month != (*p_data)[n].month) {
			printf("...   ");
			month++;
		}
		printf("%s", months[month]);
		/* Check if rainfall value for given month is validated */
		if ((*p_data)[n].validated == 'N') {
			printf("*  ");
		} else {
			printf("   ");
		}
		/* Move to next row and hence next month in dataset */
		n++;
		month++;
	}
	/* If data does not end at December, fill with "..." */
	while (month <= LAST_MONTH) {
		printf("...   ");
		month++;
	}
	printf("\n");
}

/* Find first year in dataset where required month can be found
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   req_month = required month to check for existence in dataset */
int
first_year(int num_rows, row_t (*p_data)[], int req_month) {
	int n;
	for (n=START; n<num_rows; n++) {
		/* Only return year of where the earliest match was found if it exists
		   in dataset */
		if ((*p_data)[n].month == req_month) {
			return (*p_data)[n].year;
		}
	}
	return NO_YEAR_FOUND;
}

/* Find last year in dataset where required month can be found
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   req_month = required month to check for existence in dataset */
int
last_year(int num_rows, row_t (*p_data)[], int req_month) {
	int n, end = num_rows-1;
	for (n=end; n>=START; n--) {
		/* Only return year of where the latest match was found if it exists
		   in dataset */
		if ((*p_data)[n].month == req_month) {
			return (*p_data)[n].year;
		}
	}
	return NO_YEAR_FOUND;
}

/* Calculate summation of rainfall values for given month across all years
   to later compute average for given month across all years
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   req_month = required month to check for existence in dataset */
double
sum_calc(int num_rows, row_t (*p_data)[], int req_month) {
	int n;
	double sum = 0;
	for (n=START; n<num_rows; n++) {
		if ((*p_data)[n].month == req_month) {
			sum += (*p_data)[n].rainfall;
		}
	}
	return sum;
}

/* Calculate number of rainfall values for given month recorded across all
   years
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   req_month = required month to check for existence in dataset */
int
num_values_calc(int num_rows, row_t (*p_data)[], int req_month) {
	int n, value_count = 0;
	for (n=START; n<num_rows; n++) {
		if ((*p_data)[n].month == req_month) {
			value_count++;
		}
	}
	return value_count;
}

/* Prints output for Stage 2
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   mean_vals = array containing mean values of rainfall for each month over all
   years in dataset */
void
S2_output(int num_rows, row_t (*p_data)[], double mean_vals[]) {
	int month, year_first, year_last, value_count;
	double sum;
	for (month=FIRST_MONTH; month<=LAST_MONTH; month++) {
		year_first = first_year(num_rows, p_data, month);
		year_last = last_year(num_rows, p_data, month);
		/* No found value for given month across all years, average = value
		   count = 0 */
		if (!year_first && !year_last) {
			printf("S2, %s, %2d values\n", months[month], 0);
			mean_vals[month] = 0;
			continue;
		}
		sum = sum_calc(num_rows, p_data, month);
		value_count = num_values_calc(num_rows, p_data, month);
		/* Apply formula for mean to calculate mean for every month across all
		   years */
		mean_vals[month] = sum/value_count;
		printf("S2, %s, %2d values, %d-%d, mean of %5.1lfmm\n",
			months[month], value_count, year_first, year_last,
			mean_vals[month]);
	}
}

/* Calculates only summation expression for Kendall's Tau
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   req_month = required month to check for existence in dataset */
int
tao_sum_calc(int num_rows, row_t (*p_data)[], int req_month) {
	int n, m, sum = 0;
	/* First sum in given formula */
	for (n=START; n<num_rows; n++) {
		if ((*p_data)[n].month == req_month) {
			/* Second sum in given formula */
			for (m=n+1; m<num_rows; m++) {
				if ((*p_data)[m].month == req_month) {
					if ((*p_data)[n].rainfall < (*p_data)[m].rainfall) {
						/* Delta(r_i, r_j) = +1 if r_i < r_j */
						sum++;
					} else if ((*p_data)[n].rainfall > (*p_data)[m].rainfall) {
						/* Delta(r_i, r_j) = -1 if r_i > r_j */
						sum--;
					} else {
						/* Delta(r_i, r_j) = 0 if r_i = r_j */
						sum += 0;
					}
				}
			}
		}
	}
	return sum;
}

/* Prints output for Stage 3
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset */
void
S3_output(int num_rows, row_t (*p_data)[]) {
	int month, sum, year_first, year_last, value_count;
	double tau_val;
	for (month=FIRST_MONTH; month<=LAST_MONTH; month++) {
		year_first = first_year(num_rows, p_data, month);
		year_last = last_year(num_rows, p_data, month);
		/* No found value for given month across all years, can't calculate
		   Kendall's Tau */
		if (!year_first && !year_last) {
			printf("S3, %s, %2d values\n", months[month], 0);
			continue;
		}
		sum = tao_sum_calc(num_rows, p_data, month);
		value_count = num_values_calc(num_rows, p_data, month);
		/* Cannot report Kendall's Tau for less than two instances of given
		   month in dataset. Already dealt with 0 value case in if statement
		   above */
		if (value_count == 1) {
			printf("S3, %s, %2d values\n", months[month], value_count);
		} else {
			/* Rearranged formula for Kendall's Tau */
			tau_val = ((double)NUMER/(value_count*(value_count-1)))*sum;
			printf("S3, %s, %2d values, %d-%d, tau of %5.2lf\n",
				months[month], value_count, year_first, year_last, tau_val);
		}
	}
}

/* Find highest rainfall value between all values in a given year as well as
   highest average across all months. Used to find scale for bar plot
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   year = year required to plot bar plot for
   mean_vals = array containing mean values of rainfall for each month over all
   years in dataset */
double
max_rain_point(int num_rows, row_t (*p_data)[], int year, double mean_vals[]) {
	int n, month;
	double max = 0;
	for (n=START; n<num_rows; n++) {
		if ((*p_data)[n].year == year) {
			/* Find highest rainfall value in the specified year */
			if ((*p_data)[n].rainfall > max) {
				max = (*p_data)[n].rainfall;
			}
		}
	}
	/* Check if true max value exists in the averages array from Stage 2 */
	for (month=FIRST_MONTH; month<=LAST_MONTH; month++) {
		if (mean_vals[month] > max) {
			max = mean_vals[month];
		}
	}
	return max;
}

/* Initialise all values for each month in array to 0 to prevent bar plot from
   being generated until correct row representing the rainfall value for a given
   month is reached
   Parameters:
   p_arr = pointer to the array below
   arr = the array to store YES/NO values
   length_arr = length of the array allowing for months to start from index 1 */
void
init_arr(int *p_arr, int arr[], int length_arr) {
	for (p_arr=arr; p_arr<arr+length_arr; p_arr++) {
		*p_arr = NO;
	}
}

/* Find value of rainfall for given year and month
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   year = year required to plot bar plot for
   month = month required to plot bar plot for */
double
rain_given_year_month(int num_rows, row_t (*p_data)[], int year, int month) {
	int n;
	double rainfall = 0;
	for (n=START; n<num_rows; n++) {
		if ((*p_data)[n].year == year) {
			if ((*p_data)[n].month == month) {
				rainfall = (*p_data)[n].rainfall;
			}
		}
	}
	return rainfall;
}

/* Generate bar plot for the year given as input by first removing additional
   rows
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   year = year required to plot bar plot for
   print_check_arr = the array to check for whether to start printing bar plot
   for a month
   scale_factor = the scale factor used to make maximum height of 24 rows for
   the bar plot
   mean_vals = array containing mean values of rainfall for each month over all
   years in dataset
   last_two_digits = array representing the last two digits of year required to
   plot bar plot for
   max = highest rainfall value in year the bar plot is being generated for */
void
print_bar_plot(int num_rows, row_t (*p_data)[], int year, int print_check_arr[],
		int scale_factor, double mean_vals[], char last_two_digits[],
		double max) {
	/* v represents the numbered labelled rows on the vertical axis */
	int v, start_plot = NO;
	char *p_two_digits;		/* Pointer to string of last two digits of a year */
	p_two_digits = last_two_digits;
	/* Loop from highest value of v, subtracting the scale factor at every row
	   until final row reached */
	for (v = scale_factor*MAX_HEIGHT; v>LAST_ROW; v-= scale_factor) {
		if (!start_plot) {
			/* Remove additional irrelevant rows from initial value of v in
			   which no data will be plotted */
			while (!(v-scale_factor < max && max <= v)) {
				v-= scale_factor;
			}
			start_plot = YES;
		}
		print_data_bar_plot(num_rows, p_data, year, v, print_check_arr,
			scale_factor, mean_vals, p_two_digits);
	}
}

/* Generate the actual plot for the bar plot
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   year = year required to plot bar plot for
   v = row label in the bar plot
   print_check_arr = the array to check for whether to start printing bar plot
   for a month  
   scale_factor = the scale factor used to make maximum height of 24 rows for
   the bar plot
   mean_vals = array containing mean values of rainfall for each month over all
   years in dataset
   p_two_digits = pointer to char array that represents the last two digits of
   year required to plot bar plot for */
void
print_data_bar_plot(int num_rows, row_t (*p_data)[], int year, int v,
		int print_check_arr[], int scale_factor, double mean_vals[],
		char *p_two_digits) {
	int month;
	double rainfall;
	/* Base for start of every row */
	printf("%5d | ",v);
	for (month=FIRST_MONTH; month<=LAST_MONTH; month++) {
		rainfall = rain_given_year_month(num_rows, p_data, year, month);
		/* Check value of rainfall lies within required range to start
		   plotting */
		if (v-scale_factor < rainfall && rainfall <= v) {
			print_check_arr[month] = YES;
		}
		if (print_check_arr[month] && v-scale_factor < mean_vals[month]
				&& mean_vals[month] <= v) {
			/* Plot both average and rainfall on same row */
			printf("*%s* ", p_two_digits);
		} else if (print_check_arr[month]) {
			/* Plot only rainfall */
			printf(" %s  ", p_two_digits);
		} else if (v-scale_factor < mean_vals[month]
				&& mean_vals[month] <= v) {
			/* Plot only average */
			printf("**** ");
		} else {
			/* Otherwise plot nothing */
			printf("     ");
		}
	}
	printf("\n");
}
	
/* Prints second last row of bar plot for given year */
void
print_second_last_row(void) {
	int month;
	printf("%5d ", 0);
	for (month=FIRST_MONTH; month<=LAST_MONTH; month++) {
		if (month==FIRST_MONTH) {
			printf("+-----");
		} else {
			printf("+----");
		}
	}
}

/* Prints last row of bar plot for given year */
void
print_last_row(void) {
	int month;
	printf("      ");
	for (month=FIRST_MONTH; month<=LAST_MONTH; month++) {
		if (month==FIRST_MONTH) {
			printf("  %s ",months[month]);
		} else {
			printf(" %s ",months[month]);
		}
	}
	printf("\n");
}

/* Prints output for Stage 4
   Parameters:
   num_rows = buddy variable for length of array of structs (data)
   p_data = pointer to array of structs - allows access to individual
   characteristics in each row of dataset
   argc = count of the number of strings that were on the command line that
   executed the program
   argv = pointer to char array that contains pointers to the above strings
   mean_vals = array containing mean values of rainfall for each month over all
   years in dataset */
void
S4_output(int num_rows, row_t (*p_data)[], int argc, char *argv[],
		double mean_vals[]) {
	int command_line_arg,				/* Counter for the number of years
											specified as arguments on the
											command line */ 
	year, scale_factor, print_check_arr[LEN_MONTHS],
		*p_check_arr;
	double max;
	/* Create bar plot for every command line argument given as input */
	for (command_line_arg = FIRST_ARG; command_line_arg<argc;
			command_line_arg++) {
		/* Convert string year given in input to int year */
		year = atoi(argv[command_line_arg]);
		/* String of last two digits of given year required for bar plot */
		char last_two_digits[] = {argv[command_line_arg][SECOND_LAST],
			argv[command_line_arg][LAST], '\0'};
		max = max_rain_point(num_rows, p_data, year, mean_vals);
		/* Apply ceiling function to find smallest integer greater than or equal
		   to highest point in bar plot divided by max bar plot height (24) */
		scale_factor = ceil(max/MAX_HEIGHT);
		printf("S4, %d max is %5.1lf, scale is %d\n", year, max, scale_factor);
		p_check_arr = print_check_arr;
		init_arr(p_check_arr, print_check_arr, LEN_MONTHS);
		print_bar_plot(num_rows, p_data, year, print_check_arr, scale_factor,
			mean_vals, last_two_digits, max);
		print_second_last_row();
		printf("+\n");
		print_last_row();
		printf("\n");
	}
}

/* End of Project 2 source code. Programming is fun! (please give me additional
   marks <3) */