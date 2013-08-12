# LaTeX2HTML 99.2beta6 (1.42)
# Associate labels original text with physical files.


$key = q/fig:autoimd-diagram/;
$external_labels{$key} = "$URL/" . q|ug.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:autoimd-GUI/;
$external_labels{$key} = "$URL/" . q|ug.html|; 
$noresave{$key} = "$nosave";

$key = q/par:customize/;
$external_labels{$key} = "$URL/" . q|ug.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:simsettings-GUI/;
$external_labels{$key} = "$URL/" . q|ug.html|; 
$noresave{$key} = "$nosave";

1;


# LaTeX2HTML 99.2beta6 (1.42)
# labels from external_latex_labels array.


$key = q/fig:autoimd-diagram/;
$external_latex_labels{$key} = q|1|; 
$noresave{$key} = "$nosave";

$key = q/fig:autoimd-GUI/;
$external_latex_labels{$key} = q|2|; 
$noresave{$key} = "$nosave";

$key = q/par:customize/;
$external_latex_labels{$key} = q|4|; 
$noresave{$key} = "$nosave";

$key = q/fig:simsettings-GUI/;
$external_latex_labels{$key} = q|3|; 
$noresave{$key} = "$nosave";

1;

