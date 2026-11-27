"""Verify the sliding friction assigned to MuJoCo contacts for a resting box."""

import json
import math

import mujoco


def check_contact(surface_mu, explicit_pair):
    pair = ""
    if explicit_pair:
        pair = (
            '<contact><pair geom1="surface" geom2="part" '
            f'friction="{surface_mu} {surface_mu} .005 .0001 .0001"/></contact>'
        )
    xml = f"""
        <mujoco>
          <option timestep=".002"/>
          <worldbody>
            <geom name="surface" type="plane" size="1 1 .01"
                  friction="{surface_mu} .005 .0001"/>
            <body pos="0 0 .025">
              <freejoint/>
              <geom name="part" type="box" size=".025 .025 .025"
                    mass=".06" friction="1 .005 .0001"/>
            </body>
          </worldbody>
          {pair}
        </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 100)

    if data.ncon == 0:
        raise AssertionError("Expected the box to rest on the surface")
    expected = surface_mu if explicit_pair else max(1.0, surface_mu)
    sliding_pairs = sorted({tuple(contact.friction[:2]) for contact in data.contact})
    for coefficients in sliding_pairs:
        if not all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
                   for value in coefficients):
            raise AssertionError((surface_mu, explicit_pair, coefficients, expected))
    return {
        "surface_mu": surface_mu,
        "explicit_pair": explicit_pair,
        "contacts": data.ncon,
        "tangential_mu_pairs": sliding_pairs,
    }


if __name__ == "__main__":
    print(json.dumps({
        "mujoco_version": mujoco.__version__,
        "measurement": "dimensionless solver contact parameters",
        "box_mass_kg": 0.06,
        "box_mu": 1.0,
        "timestep_s": 0.002,
        "steps": 100,
        "cases": [check_contact(mu, use_pair)
                  for use_pair in (False, True) for mu in (0.2, 0.7)],
    }, indent=2))
