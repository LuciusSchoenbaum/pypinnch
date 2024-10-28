//
// Created by Lucius Schoenbaum on 8/26/23.
//

#ifndef INC_OUTPUT_PYPINNCH_H
#define INC_OUTPUT_PYPINNCH_H

#include <stdlib.h>
#include <stdio.h>




// (Deprecated)
//
// (Internal Method)
//
// Parse labels, string of the form
// <inputs>; <outputs>
// or
// ; <outputs>
// or
// <outputs>
// where <inputs> and <outputs> are comma-separated lists of labels.
// Some typical examples: "x; u" "x, y; f" "f" "; u" "x, y, z; u, v, T, P".
// All whitespace is insignificant. Don't use any weird whitespace characters.
// If you don't use commas nothing breaks internally, but your code breaks.
//
// - labels is not modified.
// - indim*, outdim* is modified.
// - lbl* is modified, malloc'd.
//
// Result: indim, outdim are input, output dimensions respectively,
// and lbl is an array of strings containing the labels, in the order they are received.
//
// Returns: -1 on error, 0 on success.
// If 0, caller must free array lbl and its elements, otherwise not.
//
int pypinnch_internal_parse_labels___old(const char * labels, char *** lbl, int * indim, int * outdim) {
#define len_wssc 6
#define max_sz_lb 32
    char wssc[len_wssc] = " \t\n\v;,";
    // first pass.
    int len = 0;
    int nsc = 0;
    int nc1 = 0;
    int nc2 = 0;
    int I, J, i, j;
    int err_post_malloc = 0;
    int goodbefore = 0;
    int goodafter = 0;
    while (labels[len] != '\0') {
        // look for good characters
        j = 0;
        while (j < len_wssc) {
            if (labels[len] == wssc[j]) {
                // it's a bad character
                break;
            }
            ++j;
        }
        if (j == len_wssc) {
            // it's a good character
            if (nsc == 0)
                goodbefore = 1;
            else
                goodafter = 1;
        }
        if (labels[len] == ';') {
            // there is only one semicolon
            if (nsc == 1) return -1;
            ++nsc;
        }
        else if (labels[len] == ',') {
            if (nsc == 1) ++nc2;
            else ++nc1;
        }
        ++len;
        // safety
        if (len > 200) return -1;
    }
    // set *indim, *outdim
    if (nsc == 0) {
        *indim = 0;
        if (goodbefore)
            *outdim = nc1+1;
        else
            *outdim = 0;
    }
    else {
        if (goodbefore)
            *indim = nc1+1;
        else
            *indim = 0;
        if (goodafter)
            *outdim = nc2+1;
        else
            *outdim = 0;
    }
    int len_lbl = *indim + *outdim;
    // memory-alloc lbl
    *lbl = (char**)malloc((len_lbl)*sizeof(char*));
    for (i = 0; i < len_lbl; ++i) {
        (*lbl)[i] = (char*)malloc(max_sz_lb*sizeof(char));
    }
    i = 0;
    I = 0;
    // populate lbl
    while (i < len_lbl) {
        // goal: set I
        // means: walk from I until cross bad --> good
        while (err_post_malloc == 0) {
            j = 0;
            while (j < len_wssc) {
                if (labels[I] == wssc[j]) {
                    // it's a bad character
                    ++I;
                    break;
                }
                ++j;
            }
            if (j == len_wssc) {
                // it's a good character
                break;
            }
            // safety
            if (I == len) err_post_malloc = -1;
        }
        // know: I is set
        // goal: set J, the next bad
        // means: walk from I until cross good --> bad or off the edge
        J = I+1;
        while (err_post_malloc == 0) {
            if (J == len) break;
            j = 0;
            while (j < len_wssc) {
                if (labels[J] == wssc[j]) {
                    // it's a bad character
                    break;
                }
                ++j;
            }
            if (j < len_wssc) {
                // it's a bad character
                break;
            }
            else {
                // it's a good character
                ++J;
            }
        }
        if (err_post_malloc) break;
        // write
        for (j = 0; j < J - I; ++j) {
            (*lbl)[i][j] = labels[I+j];
        }
        (*lbl)[i][J-I] = '\0';
        // swing to next write
        I = J;
        ++i;
    }
    if (err_post_malloc) {
        // free memory
        for (i = 0; i < len_lbl; ++i) {
            free((*lbl)[i]);
        }
        free(*lbl);
        return -1;
    }
    return 0;
#undef len_wssc
#undef max_sz_lb
}






// (Internal Method)
//
// Parse labels, string of the form
// <inputs>; <outputs>
// where <inputs> and <outputs> are comma-separated lists of labels.
// Some typical examples: "x, t; u" "x, y, t; f" "f" "t; u" "x, y, z; u, v, T, P".
// (time.) Time is treated specially and always denoted "t". "t" cannot
// be an output label. You can place time anywhere in the input list
// but if the labels are exported back out, it will always appear last.
// (other labels.) Labels cannot contain underscores (_), hyphens (-),
// or whitespace characters. Other punctuation characters, which are
// used for various purposes by various utilities, are not recommended.
// A safe option is to stick to A-Z a-z 0-9. A label should be about the size
// of a normal English word, not larger than 32 characters.
//
//
// - labels is not modified.
// - indim*, outdim*, with_t* is modified.
// - lbl* is modified, malloc'd.
//
// Result: indim, outdim are input, output dimensions respectively,
// not counting time. with_t is 1 if time is an input, 0 if not.
// lbl is an array of strings containing the labels, in the order they
// are received, not including time.
//
// Returns: -1 on error, 0 on success.
// If 0, caller must free array lbl and its elements, otherwise not.
//
int pypinnch_internal_parse_labels(const char * labels, char *** lbl, int * indim, int * outdim, int * with_t) {
#define len_wssc 6
#define len_forb 3
#define max_sz_lb 32
    char wssc[len_wssc] = " \t\n\v;,";
    char forb[len_forb] = "_-";
    int len = 0;
    int nsc = 0;
    int nc1 = 0;
    int nc2 = 0;
    int i_time = -1;
    int I, J, i, j;
    int err_post_malloc = 0;
    int goodbefore = 0;
    int goodafter = 0;
    char tmp;
    // first pass.
    *with_t = 0;
    while (labels[len] != '\0') {
        // look for good characters
        j = 0;
        while (j < len_wssc) {
            if (labels[len] == wssc[j]) {
                // it's a bad character
                break;
            }
            ++j;
        }
        if (j == len_wssc) {
            // it's a good character
            if (nsc == 0)
                goodbefore = 1;
            else
                goodafter = 1;
        }
        j = 0;
        while (j < len_forb) {
            if (labels[len] == forb[j]) {
                break;
            }
            ++j;
        }
        if (j < len_forb) {
            // it's a forbidden character
            return -1;
        }
        if (labels[len] == ';') {
            // there is only one semicolon
            if (nsc == 1) return -1;
            ++nsc;
        }
        else if (labels[len] == ',') {
            if (nsc == 1) ++nc2;
            else ++nc1;
        }
        else if (labels[len] == 't') {
            // number of tests passed
            I = 0;
            if (len == 0) ++I;
            else {
                j = 0;
                while (j < len_wssc) {
                    if (labels[len-1] == wssc[j]) {
                        // maybe start of time
                        break;
                    }
                    ++j;
                }
                if (j < len_wssc) ++I;
            }
            if (I == 1) {
                tmp = labels[len + 1];
                if (tmp == '\0') ++I;
                else {
                    j = 0;
                    while (j < len_wssc) {
                        if (tmp == wssc[j]) {
                            // end of time, or tmp is forbidden
                            // if forbidden it will be caught later
                            break;
                        }
                        ++j;
                    }
                    if (j < len_wssc) ++I;
                }
            }
            if (I == 2) {
                if (nsc == 0) {
                    // time appears as input
                    *with_t = 1;
                    i_time = len;
                }
                else {
                    // time appears as output
                    return -1;
                }
            }
        }
        ++len;
        // safety
        if (len > 512) return -1;
    }
    if (nsc == 0) return -1;
    // with_t is either 0 or 1.
    // len is length of labels.
    // nsc is 1.
    // set *indim, *outdim
    if (goodbefore)
        *indim = nc1+1-*with_t;
    else
        return -1;
    if (goodafter)
        *outdim = nc2+1;
    else
        return -1;
    int len_lbl = *indim + *outdim;
    // memory-alloc lbl
    *lbl = (char**)malloc((len_lbl)*sizeof(char*));
    for (i = 0; i < len_lbl; ++i) {
        (*lbl)[i] = (char*)malloc(max_sz_lb*sizeof(char));
    }
    i = 0;
    I = 0;
    // populate lbl
    while (i < len_lbl) {
        // goal: set I
        // means: walk from I until cross bad --> good
        while (err_post_malloc == 0) {
            j = 0;
            while (j < len_wssc) {
                if (labels[I] == wssc[j]) {
                    // it's a bad character
                    ++I;
                    break;
                }
                ++j;
            }
            if (j == len_wssc) {
                // it's a good character
                break;
            }
            // safety
            if (I == len) err_post_malloc = -1;
        }
        // know: I is set
        // goal: set J, the next bad
        // means: walk from I until cross good --> bad or off the edge
        J = I+1;
        while (err_post_malloc == 0) {
            if (J == len) break;
            j = 0;
            while (j < len_wssc) {
                if (labels[J] == wssc[j]) {
                    // it's a bad character
                    break;
                }
                ++j;
            }
            if (j < len_wssc) {
                // it's a bad character
                break;
            }
            else {
                // it's a good character
                ++J;
            }
        }
        if (err_post_malloc) break;
        // write if label is not time
        if (I != i_time) {
            for (j = 0; j < J - I; ++j) {
                (*lbl)[i][j] = labels[I+j];
            }
            (*lbl)[i][J-I] = '\0';
            ++i;
        }
        // swing to next write
        I = J;
    }
    if (err_post_malloc) {
        // free memory
        for (i = 0; i < len_lbl; ++i) {
            free((*lbl)[i]);
        }
        free(*lbl);
        return -1;
    }
    return 0;
#undef len_wssc
#undef len_forb
#undef max_sz_lb
}



// (Internal Method)
//
// Generate a data file with rows in format
// <timestep> <time> <delta>
// and a row for each timestep called.
// Called during an output event timestep.
void pypinnch_internal_time(char * fslabels, int len_fslabels, int ti, double t) {
    char filename[len_fslabels+7];
    char line[256];
    char mode[2] = "w";
    // If the time counter is not zero, append to the file.
    if (ti != 0) mode[0] = 'a';
    sprintf(filename, "%s.t.dat", fslabels);
    FILE*fp = fopen(filename, mode);
    sprintf(line, "%d %e\n", ti, t);
    fputs(line, fp);
    fflush(fp);
    fclose(fp);
}



// (Internal Method)
//
// Generates a string that can be used
// in a Unix filename in place of a labels string.
// Examples:
// "x, y; u" ---> x-y--u
// "x, v; f" ---> x-v--f
// "x" ---> --x
// Heuristically, the symbol -- replaces a semicolon and
// a dash - replaces a comma.
int pypinnch_internal_labels2handle(
    char ** lbl,
    int indim,
    int outdim,
    int with_t,
    char * filename
) {
    int i;
    int nchar = 0;
    for (i = 0; i < indim; ++i) {
        nchar += sprintf(filename+nchar, "%s-", lbl[i]);
    }
    if (with_t)
        nchar += sprintf(filename+nchar, "t-");
    nchar += sprintf(filename+nchar, "-");
    for (i = 0; i < outdim-1; ++i) {
        nchar += sprintf(filename+nchar, "%s-", lbl[indim+i]);
    }
    nchar += sprintf(filename+nchar, "%s", lbl[indim+outdim-1]);
    return nchar;
}








/*******************************************************/
/* User Methods */







/*
 * Call once (at init, but it doesn't matter) for each output.
 * This produces a TOML metadata file.
 *
 * For ranges pass an array, for example, {1.2, 2.3, 3.4, 4.5, 5.6},
 * containing pairs of doubles describing outermost ranges
 * for variables, including both inputs and outputs to solver domain,
 * but not including time.
 * - If you wish to skip a range for some reason, state that range as 0.0, 0.0
 *   or nan, nan.
 * The output file is in TOML format, see https://toml.io/en/
 */
void pypinnch_output_init(
    char * labels,
    const double * ranges
) {
    // > parse labels
    char ** lbl;
    int i;
    int indim, outdim, with_t;
    int err;
    err = pypinnch_internal_parse_labels(labels, &lbl, &indim, &outdim, &with_t);
    if (err != 0) return;

    char filename[2048];
    int nchar = pypinnch_internal_labels2handle(lbl, indim, outdim, with_t, filename);
    nchar += sprintf(filename+nchar, ".toml");

    char lines[2028];
    nchar = 0;

    //> write
    FILE*fp = fopen(filename, "w");
    sprintf(lines, "format = 'mv1'\nlabels = '%s'\n[ranges]\n", labels);
    fputs(lines, fp);
    int nranges = indim + outdim;
    for (i = 0; i < nranges; ++i) {
        if (ranges[2*i] < ranges[2*i+1])
            nchar += sprintf(lines + nchar, "'%s' = [%e, %e]\n", lbl[i], ranges[2 * i], ranges[2 * i + 1]);
        else
            nchar += sprintf(lines + nchar, "'%s' = false\n", lbl[i]);
    }
    fputs(lines, fp);
    fflush(fp);
    fclose(fp);

    //> clean up
    for (i = 0; i < indim+outdim; ++i) {
        free(lbl[i]);
    }
    free(lbl);
}








/*
 * Output one or more Basilisk scalar fields
 *  formatted as basic multi-variable output ("mv1" format).
 *
 * The data is stored in the form
 * <handle>.<timestep>.dat
 * where <timestep> is the current timestep at the time of call (often denoted i).
 * In addition, a .toml format file called <handle>.toml is stored,
 * containing "metadata" about the data collected.
 *
 * labels: a list of labels characterizing the output.
 * For example, 2d inputs -> 1d output might be: "x, y; u".
 * Inputs (if multiple) are separated by commas, outputs (if multiple) are
 * separated by commas, and inputs and outputs are separated by a semicolon.
 *
 * list: the scalar output(s). It is assumed that the list has length `outdim`.
 *
 * ti: integer index (time step). This is most likely the timestep counter
 * used by Basilisk.
 *
 * t: the time. Basilisk uses variable t for this.
 *
*/
void pypinnch_output_scalar(
        char * labels,
        scalar * list,
        int ti,
        double t
) {
    // > parse labels
    char ** lbl;
    int indim, outdim, with_t;
    int err, i;
    err = pypinnch_internal_parse_labels(labels, &lbl, &indim, &outdim, &with_t);
    if (err != 0) return;

    char tmp0[1024];
    char tmp[1024];
    int nchar = pypinnch_internal_labels2handle(lbl, indim, outdim, with_t, tmp0);
    pypinnch_internal_time(tmp0, nchar, ti, t);
    nchar += sprintf(tmp0+nchar, ".t%d.dat", ti);

    FILE*fp = fopen(tmp0, "w");
    foreach() {
        if (indim == 0) {
            tmp[0] = '\0';
        } else if (indim == 1) {
            sprintf(tmp, "%e ", x);
        } else if (indim == 2) {
            sprintf(tmp, "%e %e ", x, y);
        } else { // indim == 3
            sprintf(tmp, "%e %e %e ", x, y, z);
        }
        fputs(tmp, fp);
        nchar = 0;
        for (scalar s in list) {
            nchar += sprintf(tmp + nchar, "%e ", val(s));
        }
        if (outdim > 0) tmp[nchar-1] = '\n';
        fputs(tmp, fp);
    }
    fflush(fp);
    fclose(fp);

    //> clean up
    for (i = 0; i < indim+outdim; ++i) {
        free(lbl[i]);
    }
    free(lbl);
}






/*
 *
 * Print an array in "mv1" format.
 * Arr is the array, nrow is the number of rows in the output,
 * So the array should be nrow*ncol where ncol = input+output dimension.
 * For example, if labels is "x, y; f" then ncol is 3.
 *
 *
 */
void pypinnch_output_array(
        char * labels,
        double * arr,
        int nrow,
        int ti,
        double t
) {
    // > parse labels
    char ** lbl;
    int indim, outdim, with_t;
    int err, i;
    err = pypinnch_internal_parse_labels(labels, &lbl, &indim, &outdim, &with_t);
    if (err != 0) return;

    char tmp0[1024];
    char tmp[1024];
    int nchar = pypinnch_internal_labels2handle(lbl, indim, outdim, with_t, tmp0);
    pypinnch_internal_time(tmp0, nchar, ti, t);
    nchar += sprintf(tmp0+nchar, ".t%d.dat", ti);

    FILE*fp = fopen(tmp0, "w");
    for (int row = 0; row < nrow; row++) {
        nchar = 0;
        for (i = 0; i < indim + outdim; ++i) {
            nchar += sprintf(tmp + nchar, "%e ", arr[row*(indim+outdim) + i]);
        }
        if (indim+outdim > 0) tmp[nchar-1] = '\n';
        fputs(tmp, fp);
    }
    fflush(fp);
    fclose(fp);

    //> clean up
    for (i = 0; i < indim+outdim; ++i) {
        free(lbl[i]);
    }
    free(lbl);
}




/*
 *
 * Print a time series in "mv1" format.
 * > the time is the input in the labels string.
 * > arr should be an array with length equal to the number of outputs.
 * Example: "t; x" to record x as t varies.
 * Then arr is an array of length 1.
 * Example: "t; a, b, c" to record a, b, c as t varies.
 * Then arr is an array of length 3.
 *
 */
void pypinnch_output_timeseries(
        char * labels,
        double * arr,
        int ti,
        double t
) {
    // > parse labels
    char ** lbl;
    int indim, outdim, with_t;
    int err, i, nchar;
    err = pypinnch_internal_parse_labels(labels, &lbl, &indim, &outdim, &with_t);
    if (err != 0) return;

    char tmp0[1024];
    char tmp1[1024];
    pypinnch_internal_labels2handle(lbl, indim, outdim, with_t, tmp0);
    sprintf(tmp1, "%s.dat", tmp0);

    char mode[2] = "w";
    // If the time counter is not zero, append to the file.
    if (ti != 0) mode[0] = 'a';
    FILE*fp = fopen(tmp1, mode);
    nchar = 0;
    nchar += sprintf(tmp0 + nchar, "%e ", t);
    for (i = 0; i < outdim; ++i) {
        nchar += sprintf(tmp0 + nchar, "%e ", arr[i]);
    }
    tmp0[nchar-1] = '\n';
    fputs(tmp0, fp);
    fflush(fp);
    fclose(fp);

    //> clean up
    for (i = 0; i < indim+outdim; ++i) {
        free(lbl[i]);
    }
    free(lbl);
}









#endif
