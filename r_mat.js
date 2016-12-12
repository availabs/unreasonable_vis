var R = {}; // the Recurrent library

(function(global) {
  "use strict";

  // Utility fun
  function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // Random numbers utils
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) { 
      return_v = false;
      return v_val; 
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  var randf = function(a, b) { return Math.random()*(b-a)+a; }
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  var randn = function(mu, std){ return mu+gaussRandom()*std; }

  // helper function returns array of zeros of length n
  // and uses typed arrays if available
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i] = 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  // Mat holds a matrix
  var Mat = function(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  }

  Mat.prototype = {
    
    get: function(row, col) { 
      // slow but careful accessor function
      // we want row-major order
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length, `Invalid get [${row}][${col}] in (${this.n},${this.d})`);
      return this.w[ix];
    },
    
    set: function(row, col, v) {
      // slow but careful accessor function
      var ix = (this.d * row) + col;
      //console.log('test' ,this.w.length,ix)
      assert(ix >= 0 && ix < this.w.length, `Invalid set [${row}][${col}] = ${v}`);
      this.w[ix] = v; 
    },

    clone: function() {
    	var out = new Mat(this.n,this.d)
    	for(var row = 0; row < this.n; row++){
    		for(var col = 0; col < this.d; col++){
    			out.set(row, col, this.get(row,col))
    		}
    	}
    	return out;
    },

    shape: function() {
    	return [this.n, this.d]
    },

    tanh: function() {
      // tanh nonlinearity
      var m = this;
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.tanh(m.w[i]);
      }
      return out;
    },
    exp: function() {
    	var m = this
    	var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.exp(m.w[i]);
      }
      return out;
    },
    sqrt: function() {
    	var m = this
    	var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.sqrt(m.w[i]);
      }
      return out;
    },
    sigmoid: function() {
      // sigmoid nonlinearity
      var m = this;
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = sig(m.w[i]);
      }
      return out;
    },

    clip: function(min, max) {
    	var m = this
    	var out = new Mat(m.n, m.d);
    	var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = this.w[i] < min ? min : this.w[i]
        out.w[i] = this.w[i] > max ? max : this.w[i]
      }
      return out;
    },

    mult: function(m2) {
    	var m1 = this;
      var out = new Mat(m1.n, m1.d);
      if(typeof m2 === 'number'){
    		for(var i=0,n=m1.w.length;i<n;i++) {
        	out.w[i] = m1.w[i] * m2;
      	}
      	return out;
    	}
			assert(m1.w.length === m2.w.length , `Invalid Division`)
    	
      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] * m2.w[i];
      }
      return out;
    },

    dot: function(m2) {
      // multiply matrices m1 * m2
      var m1 = this;
      assert(m1 && m2, 'Must supply valid matrices')
      assert(m1.d === m2.n, `Invalid * Dimensions ${m1.d} [${m1.d},${m1.n}] * [${m2.d},${m2.n}]`);

      var n = m1.n;
      var d = m2.d;
      var out = new Mat(n,d);
      for(var i=0;i<m1.n;i++) { // loop over rows of m1
        for(var j=0;j<m2.d;j++) { // loop over cols of m2
          var dot = 0.0;
          for(var k=0;k<m1.d;k++) { // dot product loop
            dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
          }
          out.w[d*i+j] = dot;
        }
      }
      return out;
    },
    
    T: function(){
    	var m = this
			var out = new Mat(this.d, this.n)
			for(var row = 0; row < this.n; row++){
    		for(var col = 0; col < this.d; col++){
    			out.w[(this.n * col) + row] =  this.w[(this.d * row) + col]
    		}
    	}
    	return out
    },
    sum: function() {
    	var out = this.w.reduce((a,b)=>{ return a+b }, 0)
    	return out
    },

    add: function(m2) {
    	var m1 = this;
    	
    	var out = new Mat(m1.n, m1.d);
    	if(typeof m2 === 'number'){
    		for(var i=0,n=m1.w.length;i<n;i++) {
        	out.w[i] = m1.w[i] + m2;
      	}
      	return out;
    	}
      assert(m1.w.length === m2.w.length)
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] + m2.w[i];
      }
      return out;
    },

    subtract: function(m2) {
    	var m1 = this;
    	
    	var out = new Mat(m1.n, m1.d);
    	if(typeof m2 === 'number'){
    		for(var i=0,n=m1.w.length;i<n;i++) {
        	out.w[i] = m1.w[i] - m2;
      	}
      	return out;
    	}
      assert(m1.w.length === m2.w.length)
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] - m2.w[i];
      }
      return out;
    },

    divide: function(m2) {
    	var m1 = this;
      var out = new Mat(m1.n, m1.d);
      if(typeof m2 === 'number'){
    		for(var i=0,n=m1.w.length;i<n;i++) {
        	out.w[i] = m1.w[i] / m2;
      	}
      	return out;
    	}
			assert(m1.w.length === m2.w.length , `Invalid Division`)
    	
      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] / m2.w[i];
      }
      return out;
    },

    pick: function(ix) {
    	var m = this;
      // pluck a row of m with index ix and return it as col vector
      assert(ix >= 0 && ix < m.n);
      var d = m.d;
      var out = new Mat(d, 1);
      for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data
      return out;
    },
    print: function(limit) {
    	var rowLimit = limit || this.n,
    			colLimit = limit || this.d
    	for(var row = 0; row < rowLimit; row++){
    		var out = row + '\t'
    		for(var col = 0; col < colLimit; col++){

    				out += (this.w[row*this.n+col] * 10).toFixed(3) +'\t' 
    		}
    		console.log(out)
    	}
    },
    toJSON: function() {
      var json = {};
      json['n'] = this.n;
      json['d'] = this.d;
      json['w'] = this.w;
      return json;
    },
    fromJSON: function(json) {
      this.n = json.n;
      this.d = json.d;
      this.w = zeros(this.n * this.d);
      this.dw = zeros(this.n * this.d);
      for(var i=0,n=this.n * this.d;i<n;i++) {
        this.w[i] = json.w[i]; // copy over weights
      }
    }
  }

  // return Mat but filled with random numbers from gaussian
  var RandMat = function(n,d,mu,std) {
  	std = std ? std : 1
  	mu = mu ? mu : 0
    var m = new Mat(n, d);
    fillRandn(m,mu,std);
    //fillRand(m,-std,std); // kind of :P
    return m;
  }

  var Zeros = function(n,d) {
    var m = new Mat(n, d);
    return m;
  }

  var Ones = function(n,d) {
    var m = new Mat(n, d);
    for(var i=0; i < m.w.length; i++) {
      m.w[i] = 1
    }
    return m;
  }

  var ZerosLike = function(m) {
  	//console.log('test', m.n, m.d)
  	return new Mat(m.n, m.d)
  }	

  // Mat utils
  // fill matrix with random gaussian numbers
  var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } }
  var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } }

  // Transformer definitions
  var softmax = function(m) {
    var out = new Mat(m.n, m.d); // probability volume
    var maxval = -999999;
    for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

    var s = 0.0;
    for(var i=0,n=m.w.length;i<n;i++) { 
      out.w[i] = Math.exp(m.w[i] - maxval);
      s += out.w[i];
    }
    for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

    // no backward pass here needed
    // since we will use the computed probabilities outside
    // to set gradients directly on m
    return out;
  }

  var sig = function(x) {
    // helper function for computing sigmoid
    return 1.0/(1+Math.exp(-x));
  }

  var maxi = function(w) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for(var i=1,n=w.length;i<n;i++) {
      var v = w[i];
      if(v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  }

  var samplei = function(w) {
    // sample argmax from w, assuming w are 
    // probabilities that sum to one
    var r = randf(0,1);
    var x = 0.0;
    var i = 0;
    while(true) {
      x += w[i];
      if(x > r) { return i; }
      i++;
    }
    return w.length - 1; // pretty sure we should never get here?
  }

  // various utils
  // global.maxi = maxi;
  // global.samplei = samplei;
  // global.randi = randi;
  // global.softmax = softmax;
  // global.assert = assert;

  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;
  global.ZerosLike = ZerosLike;
  global.Zeros = Zeros;
  global.Ones = Ones;

  // global.forwardLSTM = forwardLSTM;
  // global.initLSTM = initLSTM;
  // global.forwardRNN = forwardRNN;
  // global.initRNN = initRNN;

  // // optimization
  // global.Solver = Solver;
  // global.Graph = Graph;
  
})(R);

//module.exports = R;