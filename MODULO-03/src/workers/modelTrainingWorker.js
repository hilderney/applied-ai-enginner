import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIFGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

const normalize = (value, min, max) => (value - min) / (max - min) || 1;

function makeContext(users, products) {
    const ages = users.map(user => user.age);
    const prices = products.map(product => product.price);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index];
        })
    );

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index];
        })
    );

    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    const productAvgNorm = Object.fromEntries(
        products.map(p => {
            const avg = ageCounts[p.name] ?
                ageSums[p.name] / ageCounts[p.name] : midAge;
            return [p.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numColors: colors.length,
        numCategories: categories.length,
        dimensions: 2 + colors.length + categories.length
    };

}

const oneHotWeighted = (index, length, weight) => {
    return tf.oneHot(index, length).cast('float32').mul(weight);
}

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price, 
            context.minPrice, 
            context.maxPrice
        ) * WEIFGHTS.price
    ]);

    const age = tf.tensor1d([
        (context.productAvgNorm[product.name] ?? 0.5) * WEIFGHTS.age
    ]);

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIFGHTS.category
    );

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIFGHTS.color
    );

    return tf.concat1d([price, age, category, color]);
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    const products = await (await fetch('/data/products.json')).json();

    const context = makeContext(users, products);

    console.log('Context for training:', context);

    context.productVectors = products.map(p => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, context).dataSync()
        }
    });

    _globalCtx = context;
    
    debugger;

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
