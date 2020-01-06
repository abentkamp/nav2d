import test from "ava";

import {
    Vector,
    Edge,
    Polygon,
    isclose,
    dot,
    cross,
    NavMesh,
} from "../src/nav2d.js";

function vector() {
    return new Vector(5, 6);
}

function edge() {
    return new Edge([0, 0], [3, 4]);
}

function polygon() {
    return new Polygon([
        [0, 0],
        [0, 12],
        [12, 0],
    ]);
}

function navmesh() {
    return new NavMesh([
        [
            [0, 0],
            [0, 12],
            [12, 0],
        ],
        [
            [12, 8],
            [12, 4],
            [16, 6],
        ],
        [
            [12, 0],
            [6, 6],
            [12, 6],
        ],
        [
            [100, 100],
            [110, 100],
            [100, 110],
        ],
    ]);
}

function big_navmesh() {
    const poly_points = [];

    for (let x = 0; x < 30; x++) {
        for (let y = 0; y < 30; y++) {
            poly_points.push([
                [x * 10, y * 10],
                [x * 10, y * 10 + 10],
                [x * 10 + 10, y * 10 + 10],
                [x * 10 + 10, y * 10],
            ]);
        }
    }

    return new NavMesh(poly_points);
}

test("isclose", t => {
    t.true(isclose(5.456, 5.45600000001));
    t.true(isclose(-0, 0));
    t.false(isclose(5.456, 5.4561));
});

test("dot", t => {
    const prod = dot(vector(), vector());
    t.true(isclose(prod, 25 + 36));
});

test("cross", t => {
    const prod = cross(vector(), vector());
    t.true(isclose(prod, 0));
});

test("vector_error", t => {
    t.throws(() => {
        new Vector("a", null);
    });
});

test("vector_sum", t => {
    t.deepEqual(vector().add(vector()), new Vector(10, 12));
    t.deepEqual(vector().add(4), new Vector(9, 10));
});

test("vector_sub", t => {
    t.deepEqual(vector().sub(vector()), new Vector(0, 0));
    t.deepEqual(vector().sub(4), new Vector(1, 2));
});

test("vector_mul", t => {
    t.deepEqual(vector().mul(vector()), new Vector(25, 36));
    t.deepEqual(vector().mul(4), new Vector(20, 24));
});

test("vector_div", t => {
    t.deepEqual(vector().div(vector()), new Vector(1, 1));
    t.deepEqual(vector().div(2), new Vector(2.5, 3));
});

test("vector_length", t => {
    t.assert(isclose(vector().length(), Math.sqrt(25 + 36)));
});

test("vector_equals", t => {
    t.true(vector().equals(vector()));
    t.false(vector().equals(new Vector(100, 100)));
});

test("vector_angle", t => {
    t.assert(isclose(new Vector(1, 0).angle(new Vector(0, 1)), Math.PI / 2));
    t.assert(isclose(new Vector(1, 0).angle(new Vector(-1, 0)), Math.PI));
    t.assert(isclose(new Vector(1, 0).angle(new Vector(1, 1)), Math.PI / 4));
    t.assert(isclose(new Vector(1, 0).angle(new Vector(0, -1)), Math.PI / 2));
    t.assert(isclose(new Vector(0, -1).angle(new Vector(1, 0)), Math.PI / 2));
});

test("vector_counterclockwise_angle", t => {
    t.assert(
        isclose(
            new Vector(1, 0).counterclockwiseAngle(new Vector(0, 1)),
            Math.PI / 2
        )
    );
    t.assert(
        isclose(
            new Vector(1, 0).counterclockwiseAngle(new Vector(-1, 0)),
            Math.PI
        )
    );
    t.assert(
        isclose(
            new Vector(1, 0).counterclockwiseAngle(new Vector(1, 1)),
            Math.PI / 4
        )
    );
    t.assert(
        isclose(
            new Vector(1, 0).counterclockwiseAngle(new Vector(0, -1)),
            Math.PI * 1.5
        )
    );
    t.assert(
        isclose(
            new Vector(0, -1).counterclockwiseAngle(new Vector(1, 0)),
            Math.PI / 2
        )
    );
});

test("edge_length", t => {
    t.assert(edge().length() == 5);
});

test("edge_on_edge", t => {
    t.true(edge().onEdge([1.5, 2]));
    t.true(new Edge([0, 0], [0, 2]).onEdge([0.000000001, 1]));
    t.false(edge().onEdge([1, 2]));
    t.false(edge().onEdge([6, 8]));
});

test("edge_parallel", t => {
    t.true(edge().parallel(edge()));
    t.true(edge().parallel(new Edge([0, 0], [-3, -4])));
    t.true(edge().parallel(new Edge([1, 0], [4, 4])));
    t.false(edge().parallel(new Edge([0, 0], [1, 0])));
});

test("edge_collinear", t => {
    t.true(edge().collinear(edge()));
    t.true(edge().collinear(new Edge([0, 0], [-3, -4])));
    t.false(edge().collinear(new Edge([1, 0], [4, 4])));
    t.false(edge().collinear(new Edge([0, 0], [1, 0])));
});

test("edge_overlap", t => {
    t.true(
        edge()
            .overlap(new Edge([1.5, 2], [6, 8]))
            .equals(new Edge([1.5, 2], [3, 4]))
    );

    t.true(
        edge()
            .overlap(edge())
            .equals(edge())
    );

    t.true(
        edge()
            .overlap(new Edge([-6, -8], [-3, -4]))
            .equals(new Edge([0, 0], [0, 0]))
    );

    t.throws(() => edge().overlap(new Edge([0, 0], [1, 0])));
});

test("construct_polygon_from_array", t => {
    new Polygon([
        [0, 0],
        [0, 12],
        [12, 0],
    ]);
    t.pass();
});

test("construct_polygon_from_object", t => {
    new Polygon([
        { x: 0, y: 0 },
        { x: 0, y: 12 },
        { x: 12, y: 0 },
    ]);
    t.pass();
});

test("construct_polygon_from_vector", t => {
    new Polygon([new Vector(0, 0), new Vector(0, 12), new Vector(12, 0)]);
    t.pass();
});

test("polygon_bounds", t => {
    t.deepEqual(polygon().bounds, [0, 0, 12, 12]);
});

test("polygon_bounds_size", t => {
    t.deepEqual(
        new Polygon([
            [10, 10],
            [10, 20],
            [15, 10],
        ]).boundsSize(),
        { x: 10, y: 10, w: 5, h: 10 }
    );
});

test("polygon_edges", t => {
    const edges = polygon().edges();
    t.deepEqual(
        edges.map(e => [e.p1, e.p2]),
        [
            [new Vector(12, 0), new Vector(0, 0)],
            [new Vector(0, 0), new Vector(0, 12)],
            [new Vector(0, 12), new Vector(12, 0)],
        ]
    );
});

test("polygon_centroid", t => {
    const centroid = polygon().centroid();
    t.deepEqual(centroid, new Vector(4, 4));
});

test("polygon_on_edge", t => {
    t.truthy(polygon().onEdge([6, 6]));
    t.falsy(polygon().onEdge([7, 7]));
    t.falsy(polygon().onEdge([3, 3]));
});

test("polygon_contains", t => {
    t.true(polygon().contains([0, 0]));
    t.true(polygon().contains([1, 0]));
    t.true(polygon().contains([1, 1]));
    t.true(polygon().contains([6, 6]));
    t.true(polygon().contains([0, 12]));
    t.true(
        new Polygon([
            [0, 0],
            [0, 12],
            [1, 0],
        ]).contains([0, 12])
    );

    t.false(polygon().contains([-1, 0]));
    t.false(polygon().contains([6.1, 6]));
    t.false(polygon().contains([0, 12.001]));
});

test("navmesh_neighbours", t => {
    const mesh = navmesh();
    const [poly1, poly2, poly3, poly4] = mesh.polygons;
    t.assert(Object.values(poly1.neighbors).length == 1);
    t.assert(Object.values(poly1.neighbors)[0].polygon === poly3);

    t.assert(Object.values(poly2.neighbors).length == 1);
    t.assert(Object.values(poly2.neighbors)[0].polygon === poly3);

    t.assert(Object.values(poly3.neighbors).length == 2);
    t.assert(Object.values(poly3.neighbors)[0].polygon === poly2);
    t.assert(Object.values(poly3.neighbors)[1].polygon === poly1);

    t.assert(Object.values(poly4.neighbors).length == 0);
});

test("navmesh_find_path", t => {
    const mesh = navmesh();

    const tests = [
        [
            [1, 1],
            [14, 6],
            [new Vector(1, 1), new Vector(14, 6)],
        ],
        [
            [2, 1],
            [8, 5],
            [new Vector(2, 1), new Vector(8, 5)],
        ],
        [[-1, 1], [14, 6], null],
        [[1, 1], [105, 105], null],
    ];

    for (const [from, to, expected] of tests) {
        const path = mesh.findPath(from, to);
        t.deepEqual(path, expected);
    }
});

test("navmesh_performance", t => {
    const start1 = Date.now();
    const mesh = big_navmesh();
    const elapsed1 = Date.now() - start1;

    const start2 = Date.now();
    const path = mesh.findPath([1, 1], [99, 99]);
    const elapsed2 = Date.now() - start2;

    t.assert(elapsed1 < 2000);
    t.assert(elapsed2 < 40);

    console.log(elapsed1, elapsed2);
});