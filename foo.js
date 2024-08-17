const fs = require('fs')
const { comp } = require('./comp')

const p = JSON.parse(fs.readFileSync('compare.json', 'utf8'))
const g = JSON.parse(fs.readFileSync('c-compare.json', 'utf8'))
const f = fs.readFileSync('./scratch/train.csv', 'utf8')
    .trim()
    .split('\n')
    .slice(1, 4)
    .map(x => x.trim().split(',').map(y => parseFloat(y)))

const y = []
let x = []
for (const row of f) {
    y.push(row.shift())
    x.push(row)
}

function transpose(matrix) {
    let out = []
    for (let j = 0; j < matrix[0].length; j++) {
        out.push([])
    }
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            out[j][i] = matrix[i][j]
        }
    }
    return out
}

x = transpose(x)

console.log(comp(p.X, g.X))
console.log(comp(p.Y, g.Y))