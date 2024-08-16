const fs = require('fs')

const p = JSON.parse(fs.readFileSync('compare.json', 'utf8'))
const g = JSON.parse(fs.readFileSync('go-compare.json', 'utf8'))

function eq(a, b) {
    if (a === 0) {
        return b === 0
    }
    let x = Math.abs(a)
    let c = 0
    while (x < 1) {
        if ((x*10) === Infinity) {
            break
        }
        x = x * 10
        c++
    }
    let y = Math.abs(b) * Math.pow(10, c)
    return x.toFixed(4) === y.toFixed(4)
}

function comp(a, b) {
    if (a.length !== b.length) {
        return false
    }
    if (a[0].length !== b[0].length) {
        return false
    }

    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < a[0].length; j++) {
            if (!eq(a[i][j], b[i][j])) {
                console.log(`[${i}][${j}] ${a[i][j]} != ${b[i][j]}`)
                return false
            }
        }
    }

    return true
}

function foo(a, b) {
    if (a.length !== b.length) {
        return false
    }
    for (let i = 0; i < a.length; i++) {
        if (!eq(a[i], b[i])) {
            console.log(`[${i}] ${a[i]} != ${b[i]}`)
            return false
        }
    }
    return true
}

// INPUT
console.log('X', foo(p.X, g.X))
console.log('Y', foo(p.Y, g.Y))

// FORWARD
console.log('Z1', comp(p.Z1, g.Z1))
console.log('A1', comp(p.A1, g.A1))
console.log('Z2', comp(p.Z2, g.Z2))
console.log('A2', comp(p.A2, g.A2))

// BACKWARD
console.log('ohY', comp(p.ohY, g.ohY))
console.log('dZ2', comp(p.dZ2, g.dZ2))
console.log('dW2', comp(p.dW2, g.dW2))
console.log('db2', eq(p.db2, g.db2), p.db2, g.db2)
console.log('dZ1', comp(p.dZ1, g.dZ1))
console.log('dW1', comp(p.dW1, g.dW1))
console.log('db1', eq(p.db1, g.db1))

// UPDATE
console.log('W1', comp(p.W1, g.W1))
console.log('b1', comp(p.b1, g.b1))
console.log('W2', comp(p.W2, g.W2))
console.log('b2', comp(p.b2, g.b2))
