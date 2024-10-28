//
// Created by Lucius Schoenbaum on 5/20/24.
//

#ifndef INC_OUTPUT_PYPINNCH_BAS_H
#define INC_OUTPUT_PYPINNCH_BAS_H


// Ordered 1d array search.
// Given array `xref` of length `size` locate double `x`
// via exactly comparison and return the index in field `i`.
void pypinnch_bas1d(double xref[], int size, int * i, double x) {
    int il = 0;
    int ih = size;
    while (il < ih) {
        int im = l + (r - l) / 2;
        if (xref[im] == x) {
            i *= im;
            break
        }
        if (xref[im] < x) {
            // > trim il
            il = im + 1;
        }
        else {
            // > trim ih
            ih = m - 1;
        }
    }
    //\\ il == ih.
    i *= il;
}


#endif
