import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from platform_learning import KINDS,ReferenceEnv,ReferenceMLP,PhaseBC,specifications,protocol,episode


class PlatformTests(unittest.TestCase):
    def test_whole_episode_splits_and_single_ood_factor(self):
        for kind in KINDS:
            p=protocol(kind,"quick"); ids=[]; seeds=[]
            for split,values in p["specs"].items():
                ids.extend(s["id"] for s in values);seeds.extend(s["seed"] for s in values)
                for s in values:
                    self.assertEqual(s["mass_scale"],1.5 if split=="ood" and kind in ("panda","allegro") else 1.)
                    self.assertEqual(s["push_N"],(12. if kind=="h1" else 8.) if split=="ood" and kind in ("go2","h1") else 0.)
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(len(seeds),len(set(seeds)))

    def test_reset_preserves_physical_home(self):
        env=ReferenceEnv("h1",specifications("h1",1,333,"test")[0])
        self.assertAlmostEqual(env.d.qpos[2],.98)
        np.testing.assert_allclose(env.home[:5],[0,0,-.4,.8,-.4],atol=.006)
        self.assertEqual(env.d.time,0.)

    def test_contract_shapes_and_finite_values(self):
        expected={"panda":(65,8),"go2":(74,12),"h1":(96,19),"allegro":(86,16)}
        for kind in KINDS:
            env=ReferenceEnv(kind,specifications(kind,1,333,"test")[0])
            self.assertEqual((len(env.observation()),len(env.hold)),expected[kind])
            self.assertTrue(np.isfinite(env.observation()).all())
            self.assertEqual(env.observation()[0],0.)

    def test_panda_gripper_unit_is_separate(self):
        env=ReferenceEnv("panda",specifications("panda",1,333,"test")[0])
        action=env.hold.copy();action[-1]=.5
        env.apply(action)
        self.assertEqual(env.d.ctrl[7],127.5)
        self.assertEqual(env.c.target[-1],.02)

    def test_reference_apply_has_no_expert_fallback(self):
        env=ReferenceEnv("go2",specifications("go2",1,333,"test")[0])
        before=env.d.qpos.copy()
        with patch.object(env.c,"step",side_effect=AssertionError("Expert called")):
            env.advance(env.hold)
        self.assertEqual(env.expert_calls,0)
        self.assertAlmostEqual(env.d.time,.01)
        self.assertFalse(np.array_equal(env.d.qpos,before))

    def test_mass_and_inertia_scaled_together(self):
        specs=specifications("panda",1,333,"test")[0]
        light=ReferenceEnv("panda",specs)
        specs["mass_scale"]=1.5;heavy=ReferenceEnv("panda",specs)
        oid=light.m.body("object").id
        self.assertAlmostEqual(heavy.m.body_mass[oid]/light.m.body_mass[oid],1.5)
        np.testing.assert_allclose(heavy.m.body_inertia[oid],1.5*light.m.body_inertia[oid])

    def test_reference_mlp_gradients_in_dynamic_dimensions(self):
        rng=np.random.default_rng(7)
        model=ReferenceMLP(3,np.zeros(9),np.ones(9),np.zeros(5),np.ones(5),width=4)
        obs,target=rng.normal(size=(6,9)),rng.normal(size=(6,5))
        _,grads=model.loss_grad(obs,target)
        for key,value in model.params.items():
            index=tuple(0 for _ in value.shape);old=value[index]
            value[index]=old+1e-6;plus=model.loss_grad(obs,target)[0]
            value[index]=old-1e-6;minus=model.loss_grad(obs,target)[0]
            value[index]=old
            self.assertAlmostEqual((plus-minus)/2e-6,grads[key][index],places=5)

    def test_saved_reference_models_preserve_decoded_action(self):
        model=ReferenceMLP(3,np.zeros(9),np.ones(9),np.arange(5),np.ones(5)*.2,width=4)
        with tempfile.TemporaryDirectory(prefix="reference-bc-") as directory:
            file=Path(directory)/"model.npz";model.save(file)
            restored=ReferenceMLP.load(file)
            np.testing.assert_array_equal(restored.action(np.ones(9)),model.action(np.ones(9)))

    def test_phase_bc_uses_only_clock_and_training_labels(self):
        data={"time_s":np.array([0.,1.,0.,1.]),"action":np.array([[0.,1.],[2.,3.],[2.,3.],[4.,5.]])}
        model=PhaseBC.fit(data,2.)
        np.testing.assert_allclose(model.action(np.r_[.25,np.ones(10)]),[2.,3.])
        np.testing.assert_array_equal(model.action(np.r_[.25,np.zeros(10)]),model.action(np.r_[.25,np.ones(10)]))

    def test_real_learner_rollout_never_queries_expert(self):
        kind="go2";spec=specifications(kind,1,333,"test")[0]
        model=PhaseBC(np.array([0.,1.]),np.tile(np.array([0.,.9,-1.8]*4),(2,1)))
        with patch.object(ReferenceEnv,"expert",side_effect=AssertionError("Expert queried")):
            result,_=episode(kind,spec,"phase_bc",model)
        self.assertTrue(result["completed"])
        self.assertEqual(result["expert_calls"],0)
        self.assertEqual(result["control_steps"],1200)


if __name__=="__main__":unittest.main()
