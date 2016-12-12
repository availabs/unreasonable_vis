'use strict'


const characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,\'\n":;?!()'.split('')


const lossFun = function lossFun(inputs, targets, hprev, Wxh, Whh, Why, bh, by, vocab_size, hidden_size) {
  /*
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  */
  
  var xs = [], 
      hs = [], 
      ys = [], 
      ps = []; 
  
  var loss = 0
 
  inputs.forEach((data,t) => {
    xs[t] = R.Zeros(vocab_size, 1) // read input into  xs\
    xs[t].set(inputs[t], 1 , 1)
    var prev = t === 0 ? hprev.clone() : hs[t-1]
    hs[t] = (Wxh.dot(xs[t]).add(Whh.dot(prev)).add(bh)).tanh()
    ys[t] = Why.dot(hs[t]).add(by)
    ps[t] = ys[t].exp().divide(ys[t].exp().sum())
    loss += -Math.log(ps[t].get(targets[t],1)) // softmax (cross-entropy loss)
  })
  
  // backward pass: compute gradients going backwards
  let [dWxh, dWhh, dWhy] = [R.ZerosLike(Wxh), R.ZerosLike(Whh), R.ZerosLike(Why)]
  let [dbh, dby] = [R.ZerosLike(bh), R.ZerosLike(by)]
  let dhnext = R.ZerosLike(hs[0])
  let oneVector = R.Ones(hs[0].n, hs[0].d)
  inputs.forEach((d,t) => {
    let r = (inputs.length-1) - t
    let dy = ps[r].clone()
    // backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    
    dy.set(targets[r], 1, dy.get(targets[r],1) - 1) 
    dWhy = dWhy.add(dy.dot(hs[r].T()))
    dby = dby.add(dy)
    
    // backprop into h
    let dh = Why.T().dot(dy).add(dhnext)
    //dhraw = (1 - hs[t] * hs[t]) * dh
    // backprop through tanh nonlinearity
    let dhraw = oneVector
      .subtract(hs[t].mult( hs[t])).mult(dh)
    
    dbh = dbh.add(dhraw)
    dWxh = dhraw.dot(xs[t].T())
    
    let trailing_index = r-1 === -1 ? inputs.length-1 : r-1
    dWhh = dhraw.dot(hs[trailing_index].T())
    dhnext = Whh.T().dot(dhraw)
  });
  //console.timeEnd('backward')
  dWxh = dWxh.clip(-5, 5);
  dWhh =  dWhh.clip(-5, 5);
  dWhy = dWhy.clip(-5, 5);
  dbh =  dbh.clip(-5, 5);
  dby =  dby.clip(-5, 5);
  
  // // clip to mitigate exploding gradients
  // console.timeEnd('lossFun')
  return [loss, dWxh, dWhh, dWhy, dbh, dby, hs[inputs.length-1]]
}

const sample = function sample (h, seed_ix, n, Wxh, Whh, Why, bh, by, vocab_size){
  /* 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  */
  //console.log('h', h, Whh, h.shape, Whh.shape)
  let x = R.Zeros(vocab_size,1)
  console.log(x.shape())
  x.set(seed_ix-1, 1,  1)
  let ixes = []
  var y,p = null
  for(var t = 0; t < n; t++) {

    // h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    // y = np.dot(Why, h) + by
    // p = np.exp(y) / np.sum(np.exp(y))
    // ix = np.random.choice(range(vocab_size), p=p.ravel())
    // x = np.zeros((vocab_size, 1))
    // x[ix] = 1
    //ixes.append(ix)
    h = Wxh.dot(x).add(Whh.dot(h)).add(bh).tanh()
    y = Why.dot(h).add(by)
    p = y.exp().divide(y.exp().sum())
    //console.log('gl', p, p.shape)
    var ix = randomChoice(p.w)
    //console.log(ix)
    x.set(ix-1,1,1)
    ixes.push(characters[ix])
  }
  return ixes.join('')
}


const mainLoop = function mainLoop (data) {
  let data_size     = data.length,
      vocab_size    = characters.length;
  
  let hidden_size   = 100, // size of hidden layer of neurons
      seq_length    = 25, // number of steps to unroll the RNN for
      learning_rate = 1e-1,
      check = 1000;

  let Wxh = R.RandMat(hidden_size, vocab_size).mult(0.01), // input to hidden
      Whh = R.RandMat(hidden_size, hidden_size).mult(0.01), // hidden to hidden
      Why = R.RandMat(vocab_size, hidden_size).mult(0.01), // hidden to output
      bh = R.Zeros(hidden_size,1), //hidden bias
      by = R.Zeros(vocab_size,1); // output bias
      // console.log(np.zerosLike(Wxh));

  let [n, p] = [0, 0]
  let [mWxh, mWhh, mWhy] = [R.ZerosLike(Wxh), R.ZerosLike(Whh), R.ZerosLike(Why)]
  let [mbh, mby] = [R.ZerosLike(bh), R.ZerosLike(by)] // memory variables for Adagrad
  let smooth_loss = -Math.log(1.0/vocab_size)*seq_length // loss at iteration 0

  let thing = 0
  var hprev = R.Zeros(hidden_size,1)
  
  while(true) { //
    // prepare inputs (we're sweeping from left to right in stehhps seq_length long)
    
    if (p+seq_length+1 >= data.length || n === 0) {
      hprev = R.Zeros(hidden_size, 1) // reset RNN memory
      p = 0 // go from start of data
    }
    var inputs = data.slice(p, p+seq_length).map(char => characters.indexOf(char))
    var targets = data.slice(p+1 ,p+seq_length+1).map(char => characters.indexOf(char))
    
    // sample from the model now and then
    if (n % check === 0) {
      // sample_ix = sample(hprev, inputs[0], 200)
      // txt = sample_ix.map(char => characters[char]).join('')
      console.log( `----\n ${sample(hprev, inputs[0], 200, Wxh, Whh, Why, bh, by, vocab_size)} \n----`)
    }

    // forward seq_length characters through the net and fetch gradient
    let loss, dWxh, dWhh, dWhy, dbh, dby
    [loss, dWxh, dWhh, dWhy, dbh, dby, hprev] = lossFun(inputs, targets, hprev, Wxh, Whh, Why, bh, by, vocab_size, hidden_size)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    //console.log(loss, dWxh, dWhh, dWhy, dbh, dby, hprev)
    n % check == 0 ? 
    console.log(`iter ${n}, loss: ${smooth_loss}`)  : ''

    // console.log('yyy', Wxh.sum(), mWhh.sum(), mWhy.sum(), bh.sum(), by.sum())
    // console.log('yyy', Wxh.shape(), mWhh.shape(), mWhy.shape(), bh.shape(), by.shape())
    mWxh = mWxh.add(dWxh.mult(dWxh))
    Wxh = Wxh.add(dWxh.mult(-learning_rate).divide(mWxh.add(1e-8).sqrt()))

    mWhh = mWhh.add(dWhh.mult(dWhh))
    Whh = Whh.add(dWhh.mult(-learning_rate).divide(mWhh.add(1e-8).sqrt()))

    mWhy = mWhy.add(dWhy.mult(dWhy))
    Why = Why.add(dWhy.mult(-learning_rate).divide(mWhy.add(1e-8).sqrt()))
 
    mbh = mbh.add(dbh.mult(dbh))
    bh = bh.add(dbh.mult(-learning_rate).divide(mbh.add(1e-8).sqrt()))

    mby = mby.add(dby.mult(dby))
    by = by.add(dby.mult(-learning_rate).divide(mby.add(1e-8).sqrt()))

    // console.log('-----------------------')
    // console.log('zzz', Wxh.sum(), mWhh.sum(), mWhy.sum(), bh.sum(), by.sum())
    
    p += seq_length // move data pointer
    n += 1 // iteration counter 
  }
}


var rand = function(min, max) {
    return Math.random() * (max - min) + min;
};
 
var randomChoice = function(weight) {
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
};
