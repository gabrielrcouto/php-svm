<?php
foreach (['vendor/autoload.php', '../vendor/autoload.php', '../../autoload.php'] as $autoload) {
    $autoload = __DIR__.'/'.$autoload;
    if (file_exists($autoload)) {
        require $autoload;
        break;
    }
}

unset($autoload);

use Svm\Svm;

// Data below [0.5, 0.5] is -1, above is 1
$data = [[0, 0], [0.5, 0.5], [0.7, 0.7], [1, 1]];
$labels = [-1, -1, 1, 1];
$svm = new Svm();
$svm->train($data, $labels);

$predictions = $svm->predict([[0.1, 0.1], [0.8, 0.8]]);
var_dump($predictions);
