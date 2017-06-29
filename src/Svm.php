<?php
namespace Svm;

class Svm
{
    protected $alpha;
    protected $b;
    protected $D;
    protected $data;
    protected $kernel;
    protected $kernelResults;
    protected $kernelType;
    protected $labels;
    protected $N;
    protected $usew_;
    protected $w;

    public function train($data, $labels, $options = [])
    {
        // we need these in helper functions
        $this->data = $data;
        $this->labels = $labels;

        // parameters
        // C value. Decrease for more regularization
        $C = array_key_exists('C', $options) ? $options['C'] : 1.0;
        // numerical tolerance. Don't touch unless you're pro
        $tol = array_key_exists('tol', $options) ? $options['tol'] : 1e-4;
        // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
        $alphatol = array_key_exists('alphatol', $options) ? $options['alphatol'] : 1e-7;
        // max number of iterations
        $maxiter = array_key_exists('maxiter', $options) ? $options['maxiter'] : 10000;
        // how many passes over data with no change before we halt? Increase for more precision.
        $numpasses = array_key_exists('numpasses', $options) ? $options['numpasses'] : 20;

        // instantiate kernel according to options. kernel can be given as string or as a custom function
        $kernel = [$this, 'linearKernel'];
        $this->kernelType = 'linear';

        if (array_key_exists('kernel', $options)) {
            if (is_string($options['kernel'])) {
              // kernel was specified as a string. Handle these special cases appropriately
                if ($options['kernel'] === 'linear') {
                    $kernel = [$this, 'linearKernel'];
                    $this->kernelType = 'linear';
                }
            }

            if (is_callable($options['kernel'])) {
                // assume kernel was specified as a function. Let's just use it
                $kernel = $options['kernel'];
                $this->kernelType = 'custom';
            }
        }

        // initializations
        $this->kernel = $kernel;
        $this->N = $N = count($data);
        $this->D = $D = count($data[0]);
        $this->alpha = array_fill(0, $N, 0);
        $this->b = 0.0;
        $this->usew_ = false; // internal efficiency flag

        // Cache kernel computations to avoid expensive recomputation.
        // This could use too much memory if N is large.
        if (array_key_exists('memoize', $options) && $options['memoize']) {
            $this->kernelResults = array_fill(0, $N, []);

            for ($i = 0; $i < $N; $i++) {
                $this->kernelResults[$i] = array_fill(0, $N, []);

                for ($j = 0; $j< $N; $j++) {
                    $this->kernelResults[$i][$j] = $kernel($data[$i], $data[$j]);
                }
            }
        }

        // run SMO algorithm
        $iter = 0;
        $passes = 0;

        while ($passes < $numpasses && $iter < $maxiter) {
            $alphaChanged = 0;

            for ($i = 0; $i < $N; $i++) {
                $Ei = $this->marginOne($data[$i]) - $labels[$i];

                if (($labels[$i] * $Ei < -$tol && $this->alpha[$i] < $C)
                    || ($labels[$i] * $Ei > $tol && $this->alpha[$i] > 0)
                ) {
                    // alpha_i needs updating! Pick a j to update it with
                    $j = $i;

                    while ($j === $i) {
                        $j = rand(0, $this->N - 1);
                    }

                    $Ej = $this->marginOne($data[$j]) - $labels[$j];

                    // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
                    $ai = $this->alpha[$i];
                    $aj = $this->alpha[$j];
                    $L = 0;
                    $H = $C;

                    if ($labels[$i] === $labels[$j]) {
                        $L = max(0, $ai + $aj - $C);
                        $H = min($C, $ai + $aj);
                    } else {
                        $L = max(0, $aj - $ai);
                        $H = min($C, $C + $aj - $ai);
                    }

                    if (abs($L - $H) < 1e-4) {
                        continue;
                    }

                    $eta = 2 * $this->kernelResult($i, $j) - $this->kernelResult($i, $i) - $this->kernelResult($j, $j);

                    if ($eta >= 0) {
                        continue;
                    }

                    // compute new alpha_j and clip it inside [0 C]x[0 C] box
                    // then compute alpha_i based on it.
                    $newaj = $aj - (($labels[$j] * ($Ei - $Ej)) / $eta);

                    if ($newaj > $H) {
                        $newaj = $H;
                    }

                    if ($newaj < $L) {
                        $newaj = $L;
                    }

                    if (abs($aj - $newaj) < 1e-4) {
                        continue;
                    }

                    $this->alpha[$j] = $newaj;
                    $newai = $ai + $labels[$i] * $labels[$j] * ($aj - $newaj);
                    $this->alpha[$i] = $newai;

                    // update the bias term
                    $b1 = $this->b - $Ei - $labels[$i] * ($newai - $ai) * $this->kernelResult($i, $i)
                             - $labels[$j] * ($newaj - $aj) * $this->kernelResult($i, $j);

                    $b2 = $this->b - $Ej - $labels[$i] * ($newai - $ai) * $this->kernelResult($i, $j)
                             - $labels[$j] * ($newaj - $aj) * $this->kernelResult($j, $j);

                    $this->b = 0.5 * ($b1 + $b2);

                    if ($newai > 0 && $newai < $C) {
                        $this->b = $b1;
                    }

                    if ($newaj > 0 && $newaj < $C) {
                        $this->b = $b2;
                    }

                    $alphaChanged++;
                }
            }

            $iter++;

            //echo 'iter: ' . $iter . ' alphaChanged: ' . $alphaChanged . PHP_EOL;

            //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
            $passes = ($alphaChanged == 0) ? $passes + 1 : 0;
        }

        // if the user was using a linear kernel, lets also compute and store the
        // weights. This will speed up evaluations during testing time
        if ($this->kernelType === 'linear') {
            // compute weights and store them
            $this->w = array_fill(0, $this->D, 0);

            for ($j = 0; $j < $this->D; $j++) {
                $s = 0.0;

                for ($i = 0; $i < $this->N; $i++) {
                    $s += $this->alpha[$i] * $labels[$i] * $data[$i][$j];
                }

                $this->w[$j] = $s;
                $this->usew_ = true;
            }
        } else {
            // okay, we need to retain all the support vectors in the training data,
            // we can't just get away with computing the weights and throwing it out

            // But! We only need to store the support vectors for evaluation of testing
            // instances. So filter here based on this.alpha[i]. The training data
            // for which this.alpha[i] = 0 is irrelevant for future.
            $newdata = [];
            $newlabels = [];
            $newalpha = [];

            for ($i = 0; $i < $this->N; $i++) {
                //console.log("alpha=%f", this.alpha[i]);
                if ($this->alpha[$i] > $alphatol) {
                    $newdata[] = $this->data[$i];
                    $newlabels[] = $this->labels[$i];
                    $newalpha[] = $this->alpha[$i];
                }
            }

            // store data and labels
            $this->data = $newdata;
            $this->labels = $newlabels;
            $this->alpha = $newalpha;
            $this->N = count($this->data);
            //console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
        }

        $trainstats = [];
        $trainstats['iters'] = $iter;

        return $trainstats;
    }

    // inst is an array of length D. Returns margin of given example
    // this is the core prediction function. All others are for convenience mostly
    // and end up calling this one somehow.
    protected function marginOne($inst)
    {
        $f = $this->b;

        // if the linear kernel was used and w was computed and stored,
        // (i.e. the svm has fully finished training)
        // the internal class variable usew_ will be set to true.
        if ($this->usew_) {
            // we can speed this up a lot by using the computed weights
            // we computed these during train(). This is significantly faster
            // than the version below
            for ($j = 0; $j < $this->D; $j++) {
                $f += $inst[$j] * $this->w[$j];
            }
        } else {
            for ($i = 0; $i < $this->N; $i++) {
                $kernel = $this->kernel;
                $f += $this->alpha[$i] * $this->labels[$i] * $kernel($inst, $this->data[$i]);
            }
        }

        return $f;
    }

    public function predictOne($inst)
    {
        return $this->marginOne($inst) > 0 ? 1 : -1;
    }

    // data is an NxD array. Returns array of margins.
    protected function margins($data)
    {
        // go over support vectors and accumulate the prediction.
        $N = count($data);
        $margins = array_fill(0, $N, 0);

        for ($i = 0; $i < $N; $i++) {
            $margins[$i] = $this->marginOne($data[$i]);
        }

        return $margins;
    }

    protected function kernelResult($i, $j)
    {
        if ($this->kernelResults) {
            return $this->kernelResults[$i][$j];
        }

        $kernel = $this->kernel;

        return $kernel($this->data[$i], $this->data[$j]);
    }

    // data is NxD array. Returns array of 1 or -1, predictions
    public function predict($data)
    {
        $margs = $this->margins($data);

        for ($i = 0; $i < count($margs); $i++) {
            $margs[$i] = $margs[$i] > 0 ? 1 : -1;
        }

        return $margs;
    }

    protected function linearKernel($v1, $v2)
    {
        $s = 0;

        for ($q = 0; $q < count($v1); $q++) {
            $s += $v1[$q] * $v2[$q];
        }

        return $s;
    }

    public function save($file)
    {
        if (file_exists($file)) {
            unlink($file);
        }

        file_put_contents($file, serialize($this));
    }

    public static function load($file)
    {
        if (! file_exists($file)) {
            throw new \Exception('File not found', 1);
        }

        return unserialize(file_get_contents($file));
    }
}
