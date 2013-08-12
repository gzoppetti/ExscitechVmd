This folder contains the items necessary to play the Identification Game in offline mode. These are:

1) pdb files of molecules to be shown. Named 0, 1, 2, ... sequentially for as many files as will be used
2) Choices.txt - file containing the names of the possible molecule categories (these will show up as button options when the game is in progress). Names may contain spaces' different names are separated by '|' characters. Be sure there is no '|' or any other character (space, newline, etc.) at the end of this file. 
   example: Atom|Nucleic Acid|Ligand
3) Sequence.txt - file specifying the order in which the pdb files are to be shown. Contains the numbers of the pdb files, separated by '|' characters. Likewise, be sure there are no additional characters at the end of this file.
   example: 0|2|5|1|2|4|3      (provided you have molecules 0.pdb through 5.pdb)
