from scipy import sparse


# Returns: sparse adjacency matrix, array storing affil name for each column
# Note that we're relying on the personID to always correspond to the row number.
def read_loc_adj_mat(infile, max_rows= None):
    affil_map = {}  # label --> position
    affil_array = []    # same data, but position --> label

    # we'll create the matrix at the very end. Meanwhile, store (row,col) coords.
    coord_row = []
    coord_col = []

    with open(infile, 'r') as fpin:
        fpin.readline()     # header
        for line in fpin:
            personID, affil_string = line.split(",")
            personID = int(personID)
            if (personID > max_rows):
                break

            for affil in affil_string.split("|"):
                affil_num = affil_map.get(affil, None)
                if affil_num is None:
                    # give it a new one!
                    affil_num = len(affil_map)
                    affil_map[affil] = affil_num
                    affil_array.append(affil)


                # put it in the matrix
                coord_row.append(personID)
                coord_col.append(affil_num)

    matrix = sparse.csr_matrix(([1] * len(coord_row), (coord_row, coord_col)))
    return matrix
