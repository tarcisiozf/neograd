const fs = require('fs')

const p = JSON.parse(fs.readFileSync('compare.json', 'utf8'))
// const g = JSON.parse(fs.readFileSync('go-compare.json', 'utf8'))
const g = JSON.parse(fs.readFileSync('c-compare.json', 'utf8'))

function eq(a, b) {
    if (a === 0) {
        return b === 0
    }
    let x = Math.abs(a)
    let c = 0
    while (x < 1) {
        x = x * 10
        c++
    }
    let y = Math.abs(b) * Math.pow(10, c)
    return x.toFixed(5) === y.toFixed(5)
}

function comp(a, b) {
    if (a.length !== b.length) {
        console.log(`length ${a.length} != ${b.length}`)
        return false
    }
    if (a[0].length !== b[0].length) {
        console.log(`col length ${a[0].length} != ${b[0].length}`)
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
        console.log(`length ${a.length} != ${b.length}`)
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

const run = (name, fn) => {
    return { name, fn }
}

const pipeline = (...steps) => {
    for (const { name, fn } of steps) {
        const result = fn()
        console.log(name, result)
        if (!result) break
    }
}

pipeline(
    // INPUT
    run('X', () => comp(p.X, g.X)),
    run('Y', () => foo(p.Y, g.Y)),

    // FORWARD
    run('Z1', () => comp(p.Z1, g.Z1)),
    run('A1', () => comp(p.A1, g.A1)),
    run('Z2', () => comp(p.Z2, g.Z2)),
    run('A2', () => comp(p.A2, g.A2)),

    // BACKWARD
    run('ohY', () => comp(p.ohY, g.ohY)),
    run('dZ2', () => comp(p.dZ2, g.dZ2)),
    run('dW2', () => comp(p.dW2, g.dW2)),
    run('db2', () => eq(p.db2, g.db2), p.db2, g.db2),
    run('dZ1', () => comp(p.dZ1, g.dZ1)),
    run('dW1', () => comp(p.dW1, g.dW1)),
    run('db1', () => eq(p.db1, g.db1)),

    // UPDATE
    run('W1', () => comp(p.W1, g.W1)),
    run('b1', () => comp(p.b1, g.b1)),
    run('W2', () => comp(p.W2, g.W2)),
    run('b2', () => comp(p.b2, g.b2)),
)
