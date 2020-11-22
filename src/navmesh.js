import { v4 as uuidv4 } from "uuid";
import inside from "point-in-polygon";
import earcut from "earcut";
import QuadTree from "simple-quadtree";
import TinyQueue from "tinyqueue";

import { Vector, isclose, cross, dot } from "./math";

function _normalizePoint(point) {
    if (point instanceof Array) {
        return new Vector(...point);
    } else if (point instanceof Vector) {
        return point;
    } else if (point.hasOwnProperty("x") && point.hasOwnProperty("y")) {
        return new Vector(point.x, point.y);
    }
}

export class Edge {
    constructor(p1, p2) {
        this._uuid = uuidv4();
        this.p1 = _normalizePoint(p1);
        this.p2 = _normalizePoint(p2);
    }

    length() {
        return this.p1.sub(this.p2).length();
    }

    direction() {
        return this.p1.sub(this.p2);
    }

    onEdge(point) {
        point = _normalizePoint(point);
        const pointVec = this.p1.sub(point);

        if (!this.parallel(pointVec)) return false;

        // test that it's not only collinear, but falls between p1 and p2
        const direction = this.direction();
        const len = this.length();
        const component = dot(direction, pointVec) / (len * len);
        return component >= 0 && component <= 1;
    }

    parallel(other) {
        const otherDirection =
            other instanceof Vector ? other : other.p1.sub(other.p2);
        return isclose(cross(this.direction(), otherDirection), 0);
    }

    collinear(other) {
        const direction = this.direction();
        const otherVec1 = this.p1.sub(other.p1);
        const otherVec2 = this.p1.sub(other.p2);
        return (
            isclose(cross(direction, otherVec1), 0) &&
            isclose(cross(direction, otherVec2), 0)
        );
    }

    overlap(other) {
        if (!this.collinear(other)) {
            throw new Error(
                "Cannot compute overlap of two non-collinear edges."
            );
        }

        let endpoints = [];

        if (this.onEdge(other.p1)) endpoints.push(other.p1);
        if (this.onEdge(other.p2)) endpoints.push(other.p2);
        if (other.onEdge(this.p1)) endpoints.push(this.p1);
        if (other.onEdge(this.p2)) endpoints.push(this.p2);

        // enpoints can also be the an array with twice the same point,
        // which is fine as it yields a zero-length edge
        if (endpoints.length > 2) {
            endpoints = endpoints.filter(
                (p, i) => endpoints.findIndex((op) => op.equals(p)) === i
            );
        }

        if (!endpoints.length) {
            return null;
        } else if (endpoints.length == 1) {
            endpoints = [endpoints[0], endpoints[0]];
        }

        return new Edge(...endpoints);
    }

    distanceToPoint(point) {
        let squaredLength = dot(this.p2.sub(this.p1), this.p2.sub(this.p1));
        let r = dot(this.p2.sub(this.p1), point.sub(this.p1));
        r /= squaredLength;

        if (r < 0) {
            return point.sub(this.p1).length();
        } else if (r > 1) {
            return point.sub(this.p2).length();
        } else {
            return Math.sqrt(dot(point.sub(this.p1), point.sub(this.p1)) - r * r * squaredLength);
        }
    }

    equals(other) {
        return (
            (this.p1.equals(other.p1) && this.p2.equals(other.p2)) ||
            (this.p1.equals(other.p2) && this.p2.equals(other.p1))
        );
    }
}

export class Polygon {
    constructor(points) {
        this._uuid = uuidv4();
        this.points = points.map(_normalizePoint);
        this.bounds = this._computeBounds();
    }

    _computeBounds() {
        return this.points.reduce(
            (a, p) => [
                Math.min(p.x, a[0]),
                Math.min(p.y, a[1]),
                Math.max(p.x, a[2]),
                Math.max(p.y, a[3]),
            ],
            [Infinity, Infinity, -Infinity, -Infinity]
        );
    }

    edges() {
        return this.points.map(
            (point, i) =>
                new Edge(
                    i == 0
                        ? this.points[this.points.length - 1]
                        : this.points[i - 1],
                    point
                )
        );
    }

    centroid() {
        return this.points
            .reduce((acc, point) => acc.add(point), new Vector(0, 0))
            .div(this.points.length);
    }

    centroidDistance(other) {
        const centroidVector = this.centroid().sub(other.centroid());
        return centroidVector.length();
    }

    contains(point) {
        point = _normalizePoint(point);
        const polyPoints = this.points.map(this._toPointArray);
        return (
            inside(this._toPointArray(point), polyPoints) ||
            !!this.onEdge(point)
        );
    }

    _toPointArray(point) {
        return [point.x, point.y];
    }

    onEdge(point) {
        point = _normalizePoint(point);
        for (const edge of this.edges()) {
            if (edge.onEdge(point)) return edge;
        }

        return null;
    }

    touches(otherEdge) {
        for (const edge of this.edges()) {
            if (
                (edge.onEdge(otherEdge.p1) || edge.onEdge(otherEdge.p2)) &&
                edge.collinear(otherEdge)
            ) {
                return edge;
            }
        }

        return null;
    }

    distanceToPoint(point) {
        let shortest = Infinity;
        for (const edge of this.edges()) {
            let dist = edge.distanceToPoint(point);
            if (dist < shortest) {
                shortest = dist;
            }
        }
        return shortest;
    }

    boundsSize() {
        const [minx, miny, maxx, maxy] = this.bounds;
        return { x: minx, y: miny, w: maxx - minx, h: maxy - miny };
    }
}

export class NavMesh {
    constructor(polygons, costFunc = null, heuristicFunc = null, fromTolerance = 0, toTolerance = 0) {
        this._uuid = uuidv4();
        this.polygons = this._triangulate(polygons).map(
            (points) => new Polygon(points)
        );
        this.costFunc = costFunc;
        this.heuristicFunc = heuristicFunc;
        this.fromTolerance = fromTolerance;
        this.toTolerance = toTolerance;

        // This will be used to check point collision with
        // triangles. This should be much smaller that the typical
        // size of your mesh triangles to avoid checking too many
        // triangles for collision.
        this.pointQuerySize = 0.01;

        this._buildQuadtree();
        this._buildNeighbors();
    }

    _triangulate(polygons) {
        const triangles = [];
        for (const poly of polygons) {
            const { vertices, holes } = this._flatten(poly);
            const trianglesIndices = earcut(vertices, holes);
            for (let i = 0; i < trianglesIndices.length / 3; i++) {
                const indices = trianglesIndices.slice(i * 3, i * 3 + 3);
                triangles.push(
                    indices.map(  
                        (j) => new Vector(vertices[2 * j], vertices[2 * j + 1])
                    )
                );
            }
        }
        return triangles;
    }

    _flatten(data) {
        if (data[0].x !== undefined || typeof data[0][0] === "number") {
            // Polygon without holes
            data = [data];
        }

        let result = { vertices: [], holes: [] };
        let holeIndex = 0;

        for (let i = 0; i < data.length; i++) {
            for (let point of data[i]) {
                if (point instanceof Array) {
                    result.vertices.push(...point);
                } else {
                    result.vertices.push(point.x, point.y);
                }
            }
            if (i > 0) {
                holeIndex += data[i - 1].length;
                result.holes.push(holeIndex);
            }
        }
        return result;
    }

    _buildQuadtree() {
        // Use quad tree because the naive approach of iterating
        // with two nested for loops over the polygons has performance
        // n*lon(n), which for a 30x30 grid already takes a minute.
        // This thing, for the same grid, takes 1 second, and scales linearly.
        this.qt = QuadTree(-Infinity, -Infinity, Infinity, Infinity);
        for (const poly of this.polygons) {
            this.qt.put({
                ...poly.boundsSize(),
                polygon: poly,
            });
        }
    }

    _buildNeighbors() {
        this.polygons.forEach((polygon) => (polygon.neighbors = {}));

        for (let i = 0; i < this.polygons.length; i++) {
            const poly1 = this.polygons[i];

            for (const poly2wrap of this.qt.get(poly1.boundsSize())) {
                const poly2 = poly2wrap.polygon;

                if (poly1 === poly2) continue;
                if (poly1.neighbors.hasOwnProperty(poly2._uuid)) continue;

                const portal = this._computePortal(poly1, poly2);
                if (portal !== null && portal.length() > 0) {
                    // Ensure that portal points are given in clockwise order, viewed from the centroid of the polygon
                    let [p1, p2] = this._orderClockwise(poly1.centroid(), portal.p1, portal.p2);
                    poly1.neighbors[poly2._uuid] = { polygon: poly2, portal: new Edge(p1, p2) };
                    poly2.neighbors[poly1._uuid] = { polygon: poly1, portal: new Edge(p2, p1) };
                }
            }
        }
    }

    _computePortal(poly1, poly2) {
        for (const edge1 of poly1.edges()) {
            const edge2 = poly2.touches(edge1);
            if (edge2 !== null) {
                return edge1.overlap(edge2);
            }
        }

        return null;
    }

    findPath(from, to) {
        from = _normalizePoint(from);
        to = _normalizePoint(to);

        const path = this._findPath(from, to);
        return path && this._funnel(from, to, path);
    }

    _findPath(from, to) {
        // This is the A* algorithm
        const fromPoly = this._findClosestPolygon(from, this.fromTolerance);
        const toPoly = this._findClosestPolygon(to, this.toTolerance);

        if (fromPoly === null || toPoly === null) return null;

        const frontier = new TinyQueue(
            [{ cost: 0, polygon: fromPoly }],
            (a, b) => a.cost - b.cost
        );
        const cameFrom = { [fromPoly._uuid]: null };
        const cost = { [fromPoly._uuid]: 0 };

        while (frontier.length) {
            const current = frontier.pop().polygon;

            if (current._uuid === toPoly._uuid) {
                break;
            }

            for (const { polygon: next } of Object.values(current.neighbors)) {
                const nextCost =
                    cost[current._uuid] + this._computeCost(current, next);

                if (
                    // node not yet visited
                    !cost.hasOwnProperty(next._uuid) ||
                    // this path to node has lower cost
                    nextCost < cost[next._uuid]
                ) {
                    frontier.push({
                        cost: nextCost + this._heuristic(next, toPoly),
                        polygon: next,
                    });
                    cost[next._uuid] = nextCost;
                    cameFrom[next._uuid] = current;
                }
            }
        }

        return this._reconstructPath(toPoly, cameFrom);
    }

    _computeDistance(a, b) {
        return a.centroidDistance(b);
    }

    _computeCost(a, b) {
        if (this.costFunc !== null) {
            const portal = a.neighbors[b._uuid].portal;
            return this.costFunc(a, b, portal);
        }
        return this._computeDistance(a, b);
    }

    _heuristic(poly, to) {
        if (poly._uuid == to._uuid) return 0;
        if (this.heuristicFunc !== null) return this.heuristicFunc(poly, to);
        return this._computeDistance(poly, to);
    }

    _findContainingPolygon(point) {
        const halfQuerySize = this.pointQuerySize / 2;
        const bounds = {
            x: point.x - halfQuerySize,
            y: point.y - halfQuerySize,
            w: this.pointQuerySize,
            h: this.pointQuerySize,
        };
        for (const poly of this.qt.get(bounds)) {
            if (poly.polygon.contains(point)) return poly.polygon;
        }

        return null;
    }

    _findClosestPolygon(point, searchRadius) {
        if (searchRadius < this.pointQuerySize / 2) {
            return this._findContainingPolygon(point);
        }

        let containingPoly = this._findContainingPolygon(point);
        if (containingPoly) return containingPoly;

        const bounds = {
            x: point.x - searchRadius / 2,
            y: point.y - searchRadius / 2,
            w: searchRadius,
            h: searchRadius,
        };
        let closestDistance = Infinity;
        let closestPoly = null;
        for (const poly of this.qt.get(bounds)) {
            let distance = poly.polygon.distanceToPoint(point);
            console.log(distance)
            if (distance < closestDistance) {
                closestPoly = poly;
                closestDistance = distance;
            }
        }

        return closestPoly.polygon;
    }

    _reconstructPath(to, cameFrom) {
        if (!cameFrom.hasOwnProperty(to._uuid)) {
            // Disconnected
            return null;
        }

        let current = to;

        const path = [];
        while (current !== null) {
            path.push(current);
            current = cameFrom[current._uuid];
        }

        return path.reverse();
    }

    /** Funnel algorithm, following https://medium.com/@reza.teshnizi/the-funnel-algorithm-explained-visually-41e374172d2d */
    _funnel(from, to, path) {
        if (path.length === 0) {
            throw new Error("Path cannot be empty.");
        } else if (path.length === 1) {
            return [from, to];
        }

        const tail = [from];
        const left = [];
        const right = [];

        // Initialize funnel
        const initialPortal = path[0].neighbors[path[1]._uuid].portal;
        left.push(initialPortal.p1);
        right.push(initialPortal.p2);

        // Iterate over portals
        for (let i = 1; i < path.length - 1; i++) {
            const poly = path[i];
            const nextPoly = path[i + 1];
            const portal = poly.neighbors[nextPoly._uuid].portal;
            // The portal end points are in clockwise order, viewed from the inside of the polygon.

            // Extend funnel on the left
            this._extendFunnel(tail, left, right, true, portal.p1);

            // Extend funnel on the right
            this._extendFunnel(tail, left, right, false, portal.p2);
        }

        // Close funnel to endpoint
        this._extendFunnel(tail, left, right, true, to);
        this._extendFunnel(tail, left, right, false, to);

        return tail;
    }

    _extendFunnel(tail, left, right, extendLeft, newPoint){
        const apex = tail[tail.length - 1];
        // We pretend to be in the `expandLeft` case here. Otherwise flip.
        if (!extendLeft) {
            [left, right] = [right, left];
        }

        // If `newPoint` is the end point of the left side of the funnel, skip it.
        const lastLeft = left.length === 0 ? tail[tail.length - 1] : left[left.length - 1];
        if (newPoint.equals(lastLeft)) {
            return;
        }

        // Determine angle of `apex`-`newPoint` relative to `apex`-`left[j]`
        let j = 0;
        while (j < left.length
                && this._isInClockwiseOrder(apex, newPoint, left[j], !extendLeft)) {
            j++;
        }
        // All points in `left` with index `< j` are right of `newPoint` and
        // all points in `left` with index `>= j` are left of or at the same angle as `newPoint`. 
        left.length = j; // Shrink funnel if `j < left.length`
        left.push(newPoint);
        if (j === 0) {
            // If the funnel shrunk all the way on the left, it might collapse to the right.
            // Determine how far it needs to collapse
            let k = 0;
            while (k < right.length
                    && !this._isInClockwiseOrder(apex, newPoint, right[k], !extendLeft)) {
                k++;
            }
            // All points in `right` with index `< k` are left of or at the same angle as `newPoint` and
            // all points in `right` with index `>= k` are right of `newPoint`. 
            tail.push(...right.splice(0, k)); // Collapse funnel if `k > 0`
        }
    }

    /** 
     * Are the points `p1` and `p2` in clockwise order, viewed from `origin`?
     * Checks for counter-clockwise order instead if `flip = true`.
     * Returns `false` if the angles are equal.
     */
    _isInClockwiseOrder(origin, p1, p2, flip = false) {
        if (flip) {
            [p1, p2] = [p2, p1];
        }
        const vec1 = p1.sub(origin);
        const vec2 = p2.sub(origin);
        return cross(vec1, vec2) > 0;
    }

    /** Returns the points `p1` and `p2` in clockwise order, viewed from `origin`. */
    _orderClockwise(origin, p1, p2) {
        if (this._isInClockwiseOrder(origin, p1, p2)) {
            return [p1, p2];
        } else {
            return [p2, p1];
        }
    }

}
