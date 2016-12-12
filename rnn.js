var RNN = function (vocab_size, hidden_size, seq_length) {
	
  	// test 123
  	
	this.vocab_size =	vocab_size
	this.hidden_size = 	 hidden_size// size of hidden layer of neurons
    this.seq_length = 	seq_length // number of steps to unroll the RNN for
    this.learning_rate = 1e-1
    this.iter = 			0
    
  	//the memory
  	this.Wxh = 			R.RandMat(this.hidden_size, this.vocab_size).mult(0.01) // input to hidden
    this.Whh = 			R.RandMat(this.hidden_size, this.hidden_size).mult(0.01) // hidden to hidden
    this.Why = 			R.RandMat(this.vocab_size, this.hidden_size).mult(0.01) // hidden to output
    this.bh = 			R.Zeros(this.hidden_size, 1) //hidden bias
    this.by = 			R.Zeros(this.vocab_size, 1) // output bias
   	
    
  	// memory variables for Adagrad
  	this.mWxh = 			R.ZerosLike(this.Wxh)
  	this.mWhh = 			R.ZerosLike(this.Whh) 
  	this.mWhy = 			R.ZerosLike(this.Why)
  	this.mbh = 			R.ZerosLike(this.bh)
  	this.mby = 			R.ZerosLike(this.by)
  	this.smooth_loss = 	-Math.log(1.0/vocab_size)*this.seq_length // loss at iteration 0
  	this.loss_history =  []
  	this.sloss_history = []
  	this.hprev = 		R.Zeros(hidden_size, 1)

	this.layers = ['Wxh', 'Whh', 'Why','mWxh','mWhh','mWhy']
	this.layers.forEach(d =>{

		this[d + '_graph'] =  d3.select('#display')
			.append('div')
			.attr('class', 'matrix')
			.append('svg')
	})
		
  	this.updateGraphs()
}

RNN.prototype = {
  	updateGraphs: function() {
  		var t = d3.transition()
      		.duration(750);
  		var colorScale = d3.scaleLinear()
  			.domain([-0.1, 0, 0.1])
  			.range(['#000','#888','#fff'])

  		var cellSize = 3
  		d3.select('#iter').text(this.iter)
  		d3.select('#loss').text(this.smooth_loss)

  		this.layers.forEach(d =>{

  			let data_json = this[d].toJSON();
  			console.log(d, [d3.min(data_json.w), d3.max(data_json.w)], data_json.d, data_json.n)
	  		colorScale.domain([d3.min(data_json.w), d3.mean(data_json.w), d3.max(data_json.w)])
	  		
	  		this[d + '_graph']
	  			.attr('width', cellSize*data_json.d)
	  			.attr('height', cellSize*data_json.n) 
	  			
	  		var boxes = this[d + '_graph'].selectAll('rect')
	  			.data(data_json.w)

	  		boxes
	  			.transition(t)
      				.attr('fill', (d) => colorScale(d))
	  		
	  		boxes	
	  			.enter()
	  			.append('rect')
	  				.attr('x', (d,i) => i % data_json.d * cellSize)
	  				.attr('y', (d,i) => Math.floor(i / data_json.d) * cellSize)
	  				.attr('width', cellSize)
	  				.attr('height', cellSize)
	  				.attr('fill', (d) => colorScale(d) )
	  				.on('mouseover', function (d,i) {
	  					//console.log('test', characters[i%data_json.d], Math.floor(i / data_json.d))
	  					var box = d3.select(this)
	  						//.attr('fill', 'pink')
	  						//.attr('stroke-width', '3px')
	  				})
	  				.on('mouseout', function (d) {
	  					// console.log('test', d, d3.select(this))
	  					var box = d3.select(this)
	  						//.attr('fill', colorScale(d))
	  						//.attr('stroke-width', '3px')
	  				})
	  	})
  	},

  	lossFunction: function (inputs, targets) {
  		var xs = [], 
		    hs = [], 
		    ys = [], 
		    ps = []; 
	  
	  	var loss = 0
		// forward pass	 
		inputs.forEach((data,t) => {
		    xs[t] = R.Zeros(this.vocab_size, 1) // read input into  xs\
		    xs[t].set(inputs[t], 1 , 1)
		    var prev = t === 0 ? this.hprev.clone() : hs[t-1]
		    hs[t] = (this.Wxh.dot(xs[t]).add(this.Whh.dot(prev)).add(this.bh)).tanh()
		    ys[t] = this.Why.dot(hs[t]).add(this.by)
		    ps[t] = ys[t].exp().divide(ys[t].exp().sum())
		    loss += -Math.log(ps[t].get(targets[t],1)) // softmax (cross-entropy loss)
		})
	  
	  	// backward pass: compute gradients going backwards
		let [dWxh, dWhh, dWhy] = [R.ZerosLike(this.Wxh), R.ZerosLike(this.Whh), R.ZerosLike(this.Why)]
		let [dbh, dby] = [R.ZerosLike(this.bh), R.ZerosLike(this.by)]
		let dhnext = R.ZerosLike(hs[0])
		let oneVector = R.Ones(hs[0].n, hs[0].d)
		
		// backward pass
		inputs.forEach((d,t) => {
		    let r = (inputs.length-1) - t
		    let dy = ps[r].clone()
		    dy.set(targets[r], 1, dy.get(targets[r],1) - 1) 
		    dWhy = dWhy.add(dy.dot(hs[r].T()))
		    dby = dby.add(dy)
		  
		    let dh = this.Why.T().dot(dy).add(dhnext)
		    let dhraw = oneVector
		      .subtract(hs[t].mult( hs[t])).mult(dh)
		    
		    dbh = dbh.add(dhraw)
		    dWxh = dhraw.dot(xs[t].T())
		    
		    let trailing_index = r-1 === -1 ? inputs.length-1 : r-1
		    dWhh = dhraw.dot(hs[trailing_index].T())
		    dhnext = this.Whh.T().dot(dhraw)
		});
	  	dWxh = dWxh.clip(-5, 5);
	  	dWhh =  dWhh.clip(-5, 5);
	  	dWhy = dWhy.clip(-5, 5);
	  	dbh =  dbh.clip(-5, 5);
	  	dby =  dby.clip(-5, 5);
	  	this.loss_history.push(loss)
	  	this.smooth_loss = this.smooth_loss * 0.999 + loss * 0.001
	  	this.sloss_history.push(this.smooth_loss)
	  	this.hprev= hs[inputs.length-1]
	  	this.Adagrad(dWxh, dWhh, dWhy, dbh, dby)

	},

	Adagrad: function (dWxh, dWhh, dWhy, dbh, dby) {
		
		this.mWxh = this.mWxh.add(dWxh.mult(dWxh))
	    this.Wxh = this.Wxh.add(dWxh.mult(-this.learning_rate).divide(this.mWxh.add(1e-8).sqrt()))

	    this.mWhh = this.mWhh.add(dWhh.mult(dWhh))
	    this.Whh = this.Whh.add(dWhh.mult(-this.learning_rate).divide(this.mWhh.add(1e-8).sqrt()))

	    this.mWhy = this.mWhy.add(dWhy.mult(dWhy))
	    this.Why = this.Why.add(dWhy.mult(-this.learning_rate).divide(this.mWhy.add(1e-8).sqrt()))
	 
	    this.mbh = this.mbh.add(dbh.mult(dbh))
	    this.bh = this.bh.add(dbh.mult(-this.learning_rate).divide(this.mbh.add(1e-8).sqrt()))

	    this.mby = this.mby.add(dby.mult(dby))
	    this.by = this.by.add(dby.mult(-this.learning_rate).divide(this.mby.add(1e-8).sqrt()))

	    this.iter++
	    
	},

	export: function () {
		var output = {}
		output.vocab_size = this.vocab_size
  		output.hidden_size = this.hidden_size
	    output.seq_length = this.seq_length
	    output.learning_rate = this.learning_rate
	    output.iter = this.iter
	    
	  	//the memory
	  	output.Wxh = this.Wxh.toJSON()
	    output.Whh = this.Whh.toJSON()
	    output.Why = this.Why.toJSON()
	    output.bh = this.bh.toJSON()
	    output.by = this.by.toJSON()
	   	
	    
	  	// memory variables for Adagrad
	  	output.mWxh = this.mWxh.toJSON()
	  	output.mWhh = this.mWhh.toJSON()
	  	output.mWhy = this.mWhy.toJSON()
	  	output.mbh = this.mbh.toJSON()
	  	output.mby = this.mby.toJSON()
	  	output.smooth_loss = this.smooth_loss
	  	output.loss_history = this.loss_history
	  	output.sloss_history = this.sloss_history
	  	output.hprev = this.hprev
	  	return JSON.stringify(output)
	},

	randomChoice: function (weight) {
	  var total_weight = weight.reduce(function (prev, cur, i, arr) {
	    return prev + cur;
	  });
	   
	  var random_num = rand(0, total_weight);
	  var weight_sum = 0;
	  //console.log(random_num)
	   
	  for (var i = 0; i < weight.length; i++) {
	    weight_sum += weight[i];
	    weight_sum = +weight_sum.toFixed(4);
	     
	    if (random_num <= weight_sum) {
	      return i
	    }
	  }   
	  // end of function
	},

	sample: function (seed_ix, n) {
		console.log('sample', seed_ix, n)
	  /* 
	  sample a sequence of integers from the model 
	  h is memory state, seed_ix is seed letter for first time step
	  */
	  //console.log('h', h, Whh, h.shape, Whh.shape)
	  let x = R.Zeros(this.vocab_size,1)
	  x.set(seed_ix-1, 1,  1)
	  let ixes = []
	  var y,p = null
	  for(var t = 0; t < n; t++) {

	    h = this.Wxh.dot(x).add(this.Whh.dot(this.hprev)).add(this.bh).tanh()
	    y = this.Why.dot(h).add(this.by)
	    p = y.exp().divide(y.exp().sum())
	    //console.log('gl', p, p.shape)
	    var ix = this.randomChoice(p.w)
	    //console.log(ix)
	    x.set(ix-1,1,1)
	    ixes.push(characters[ix])
	  }
	  return ixes.join('')
	}


}